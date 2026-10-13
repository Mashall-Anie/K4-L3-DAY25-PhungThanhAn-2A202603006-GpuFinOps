"""Efficiency metrics — the numbers that actually drive GPU cost.

Key teaching point (deck §5): nvidia-smi "GPU-Util %" is a *time-active* clock,
not an efficiency metric. A GPU can read 100% util while its MFU is ~20% — you
are paying the full GPU-hour for a fraction of the FLOPs you rented.
"""
from __future__ import annotations


def compute_mfu(achieved_tflops: float, peak_tflops: float) -> float:
    """Model FLOPs Utilization = achieved / peak (clamped to 0..1).

    Good training MFU is ~0.35-0.45; >0.50 is excellent. Returns 0 if peak<=0.
    """
    if peak_tflops <= 0:
        return 0.0
    return max(0.0, min(1.0, achieved_tflops / peak_tflops))


def compute_mbu(achieved_bw_tbs: float, peak_bw_tbs: float) -> float:
    """Model Bandwidth Utilization = achieved HBM BW / peak BW (clamped 0..1).

    The right metric for memory-bound decode; target ~0.60 on H100-80GB batch-1.
    """
    if peak_bw_tbs <= 0:
        return 0.0
    return max(0.0, min(1.0, achieved_bw_tbs / peak_bw_tbs))


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """FLOP / byte for a workload (the x-axis of the roofline model)."""
    if bytes_moved <= 0:
        return 0.0
    return flops / bytes_moved


def roofline_regime(intensity: float, ridge_point: float) -> str:
    """Below the ridge point a workload is memory-bound; at/above it is compute-bound.

    H100 ridge ~295 FLOP/byte (BF16). LLM decode (~1-2) is memory-bound; prefill
    (~455) is compute-bound — which is *why* prefill/decode disaggregation pays off.
    """
    return "compute-bound" if intensity >= ridge_point else "memory-bound"


def flag_util_lies(rows, util_threshold: float = 0.90, mfu_threshold: float = 0.30):
    """Return the rows where GPU-Util is high but MFU is low — money leaking.

    `rows` is an iterable of dicts each having 'gpu_util_pct' (0-100) and 'mfu' (0-1).
    These are GPUs you are billed full-rate for while they do little real compute.
    """
    out = []
    for r in rows:
        util = float(r.get("gpu_util_pct", 0)) / 100.0
        mfu = float(r.get("mfu", 0))
        if util >= util_threshold and mfu < mfu_threshold:
            out.append(r)
    return out


def idle_waste_usd(idle_hours: float, on_demand_hr: float) -> float:
    """Dollars burned by a GPU left running idle (training done, instance up)."""
    return max(0.0, idle_hours) * max(0.0, on_demand_hr)


# ---------------------------------------------------------------------------
# "Your Turn" extension 2 — MBU-driven right-sizing
#
# An idle hour tells you the GPU has nothing to do. MFU/MBU tell you something
# stronger: you bought the wrong *shape* of GPU. If a card never gets close to
# its bandwidth ceiling (MBU target) *and* never gets close to its FLOPs ceiling
# (MFU cap), the workload is neither compute- nor bandwidth-hungry — the extra
# silicon is pure rent. Buy the cheapest card that can still deliver the peak
# bandwidth and peak FLOPs the job actually demonstrated, with HBM headroom.
# ---------------------------------------------------------------------------

DEFAULT_MBU_TARGET = 0.60   # a well-fed card should clear ~60% of HBM bandwidth
DEFAULT_MFU_CAP = 0.30      # below this, the FLOPs you rent are mostly unused
SIZING_MFU_TARGET = 0.75    # the replacement card must run the job at <= 75% MFU
SIZING_MBU_TARGET = 0.60    # ... and at <= 60% MBU, so the busiest hour still fits
BW_SAFETY_MARGIN = 1.05     # require 5% bandwidth headroom on the target card


def dollars_per_gb_vram(on_demand_hr: float, hbm_gb: float) -> float:
    """Rental $/hour per GB of HBM — the price of the memory you are renting."""
    return on_demand_hr / hbm_gb if hbm_gb > 0 else 0.0


def is_overprovisioned(mfu: float, mbu: float,
                       mfu_cap: float = DEFAULT_MFU_CAP,
                       mbu_target: float = DEFAULT_MBU_TARGET) -> bool:
    """True when a GPU is far from *both* ceilings — neither compute nor BW is the limit.

    Only then is moving to a smaller/cheaper card safe: a memory-bound job at
    55% MBU is already using the bandwidth it pays for, so downgrading it would
    just make the job slower without freeing meaningful money.
    """
    return mfu < mfu_cap and mbu < mbu_target


def rightsize_gpu(
    achieved_tflops: float,
    achieved_bw_tbs: float,
    mem_peak_gb: float,
    current_type: str,
    catalog: dict,
    mbu_target: float = SIZING_MBU_TARGET,
    mfu_target: float = SIZING_MFU_TARGET,
    margin: float = BW_SAFETY_MARGIN,
) -> dict:
    """Recommend the cheapest GPU type that can still run the *observed peak* workload.

    `catalog` maps gpu_type -> row dict with on_demand_hr / hbm_gb /
    peak_tflops_fp16 / peak_bw_tbs. Sizing uses observed **peaks** (max over the
    telemetry window), never the mean: right-sizing must not clip the busiest hour.
    """
    peak_bw_need = (achieved_bw_tbs / mbu_target) * margin if mbu_target > 0 else 0.0
    peak_tflops_need = achieved_tflops / mfu_target if mfu_target > 0 else 0.0
    current_od = float(catalog[current_type]["on_demand_hr"])

    options = []
    for gtype, row in catalog.items():
        hbm = float(row["hbm_gb"])
        bw = float(row["peak_bw_tbs"])
        tf = float(row["peak_tflops_fp16"])
        od = float(row["on_demand_hr"])
        fits = hbm >= mem_peak_gb and bw >= peak_bw_need and tf >= peak_tflops_need
        options.append({
            "gpu_type": gtype, "on_demand_hr": od, "hbm_gb": hbm,
            "peak_bw_tbs": bw, "peak_tflops_fp16": tf,
            "fits": fits, "cheaper_than_current": od < current_od,
            "dollars_per_gb_vram": round(dollars_per_gb_vram(od, hbm), 4),
        })

    eligible = [o for o in options if o["fits"] and o["cheaper_than_current"]]
    best = min(eligible, key=lambda o: o["on_demand_hr"]) if eligible else None
    return {
        "current_type": current_type,
        "current_on_demand_hr": current_od,
        "bw_required_tbs": round(peak_bw_need, 3),
        "tflops_required": round(peak_tflops_need, 1),
        "hbm_required_gb": round(mem_peak_gb, 1),
        "recommended_type": best["gpu_type"] if best else current_type,
        "recommended_on_demand_hr": best["on_demand_hr"] if best else current_od,
        "monthly_savings_per_gpu": round(max(0.0, current_od - (best["on_demand_hr"] if best else current_od)) * 24 * 30, 2),
        "options": options,
    }