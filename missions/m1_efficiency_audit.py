"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py

Adds "Your Turn" extension 2: MBU-driven right-sizing. A GPU-Util lie tells you
the card is busy but unproductive; MBU/MFU peaks tell you *why* — and whether a
cheaper card could have done the same work for less rent.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics

DAYS = 30


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU (means for the lie test, peaks for sizing)
    agg = defaultdict(lambda: {
        "util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0,
        "mem_peak": 0.0, "tflops_peak": 0.0, "bw_peak": 0.0,
    })
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1
        a["mem_peak"] = max(a["mem_peak"], num(r["mem_used_gb"]))
        a["tflops_peak"] = max(a["tflops_peak"], num(r["achieved_tflops"]))
        a["bw_peak"] = max(a["bw_peak"], num(r["achieved_bw_tbs"]))

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mfu_peak": round(max(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "mbu_peak": round(max(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
            "mem_peak_gb": round(a["mem_peak"], 1),
            "tflops_peak": round(a["tflops_peak"], 1),
            "bw_peak_tbs": round(a["bw_peak"], 3),
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # --- EXTENSION 2: right-size the GPUs that never approach either ceiling ---
    rightsizing = []
    for s in summary:
        if not metrics.is_overprovisioned(s["mfu_peak"], s["mbu_peak"]):
            continue
        rec = metrics.rightsize_gpu(
            achieved_tflops=s["tflops_peak"], achieved_bw_tbs=s["bw_peak_tbs"],
            mem_peak_gb=s["mem_peak_gb"], current_type=s["gpu_type"], catalog=cat,
        )
        rec["gpu_id"] = s["gpu_id"]
        rec["mfu"] = s["mfu"]
        rec["mbu"] = s["mbu"]
        rec["dollars_per_gb_vram"] = round(
            metrics.dollars_per_gb_vram(num(cat[s["gpu_type"]]["on_demand_hr"]),
                                        num(cat[s["gpu_type"]]["hbm_gb"])), 4)
        rightsizing.append(rec)
    rightsize_monthly = sum(r["monthly_savings_per_gpu"] for r in rightsizing)
    idle_monthly = idle_waste * DAYS

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_monthly:,.0f}/month")

        print(f"\n-- Extension 2: MBU-driven right-sizing "
              f"(flag when MFU_peak < {metrics.DEFAULT_MFU_CAP:.2f} AND MBU_peak < {metrics.DEFAULT_MBU_TARGET:.2f}) --")
        print(f"{'GPU':13}{'from':6}{'to':8}{'why (peak demand)':46}{'$/mo':>9}")
        for r in sorted(rightsizing, key=lambda x: -x["monthly_savings_per_gpu"]):
            need = (f"needs >={r['bw_required_tbs']:.2f} TB/s @60% MBU, "
                    f"{r['tflops_required']:.0f} TF @75% MFU, {r['hbm_required_gb']:.0f} GB HBM")
            print(f"{r['gpu_id']:13}{r['current_type']:6}{r['recommended_type']:8}{need:46}"
                  f"${r['monthly_savings_per_gpu']:>8,.0f}")
        print(f"Right-sizing total: ${rightsize_monthly:,.0f}/month "
              f"(no card added, same jobs, cheaper silicon)")

    return {
        "summary": summary, "lies": lies,
        "idle_waste_daily": round(idle_waste, 2),
        "idle_waste_monthly": round(idle_monthly, 2),
        "rightsizing": rightsizing,
        "rightsize_monthly_savings": round(rightsize_monthly, 2),
    }


if __name__ == "__main__":
    run()
