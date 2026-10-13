"""Sustainability economics — energy and carbon as governed cost levers (deck §11).

Region selection cuts $ and carbon together; reasoning queries are an energy bomb.
"""
from __future__ import annotations

# Grid carbon intensity (gCO2 / kWh) — illustrative 2026 snapshot.
REGION_CARBON = {
    "us-east-1": 380,
    "us-west-2": 120,   # Oregon hydro
    "europe-north1": 30,  # Norway
    "europe-central2": 660,  # Poland (dirtiest)
    "us-east-wa": 90,
}
# Electricity price (USD / kWh) — illustrative.
REGION_PRICE_KWH = {
    "us-east-1": 0.12,
    "us-west-2": 0.07,
    "europe-north1": 0.09,
    "europe-central2": 0.18,
    "us-east-wa": 0.055,
}

REASONING_ENERGY_MULTIPLIER = 80.0  # deck: reasoning ~74-86x a small-model query


def wh_per_query(total_tokens: int, wh_per_1k_tokens: float = 0.30, is_reasoning: bool = False) -> float:
    """Energy for one query. Median Gemini prompt ~0.24 Wh; reasoning ~74-86x."""
    base = (total_tokens / 1000.0) * wh_per_1k_tokens
    return base * (REASONING_ENERGY_MULTIPLIER if is_reasoning else 1.0)


def carbon_g(wh: float, region: str = "us-east-1") -> float:
    """Grams CO2 for an energy amount in a region."""
    gco2_kwh = REGION_CARBON.get(region, 400)
    return (wh / 1000.0) * gco2_kwh


def energy_cost_usd(wh: float, region: str = "us-east-1") -> float:
    """Electricity cost of an energy amount in a region."""
    return (wh / 1000.0) * REGION_PRICE_KWH.get(region, 0.12)


def tokens_per_watt(total_tokens: int, wh: float, seconds: float = 1.0) -> float:
    """Energy efficiency of serving: tokens per watt (higher is better)."""
    watts = (wh * 3600.0) / seconds if seconds > 0 else 0.0
    return total_tokens / watts if watts > 0 else 0.0



# ===========================================================================
# EXTENSION 5 — carbon-aware scheduling
#
# Region choice is a FinOps lever, not a CSR footnote: the same GPU-hour can
# emit 13x more CO2e and cost 2x more electricity depending on where it runs.
# Interruptible jobs are the ones you are allowed to move, so they are the ones
# you schedule against the cleanest-and-cheapest grid.
# ===========================================================================

DEFAULT_REGION = "us-east-1"


def gpu_energy_kwh(gpu_hours: float, watts: float) -> float:
    """Energy drawn by GPU-hours at a given board power (kWh)."""
    return max(0.0, gpu_hours) * max(0.0, watts) / 1000.0


def region_score(region: str, carbon_weight: float = 0.5) -> float:
    """Blended cost+carbon score (lower is better) used to rank regions.

    carbon_weight=0 ranks on electricity price alone, 1 on carbon alone.
    """
    carbon = REGION_CARBON.get(region, 400)
    price = REGION_PRICE_KWH.get(region, 0.12)
    max_carbon = max(REGION_CARBON.values())
    max_price = max(REGION_PRICE_KWH.values())
    return carbon_weight * (carbon / max_carbon) + (1.0 - carbon_weight) * (price / max_price)


def cleanest_cheapest_region(carbon_weight: float = 0.5) -> str:
    """Region minimising the blended electricity-price + carbon score."""
    return min(REGION_CARBON, key=lambda r: region_score(r, carbon_weight))


def carbon_aware_plan(jobs, catalog, from_region: str = DEFAULT_REGION,
                      to_region: str | None = None) -> dict:
    """Move interruptible jobs to the best region and price the win.

    `jobs` is a list of workload dicts (hours_per_day/days/num_gpus/gpu_type);
    `catalog` maps gpu_type -> price-catalog row (for `watts`).
    Returns gCO2e/electricity saved plus the per-job detail behind the number.
    """
    to_region = to_region or cleanest_cheapest_region()
    moved, details = 0, []
    kwh_total = 0.0
    carbon_from = carbon_to = cost_from = cost_to = 0.0

    for j in jobs:
        if not int(float(j.get("interruptible", 0) or 0)):
            continue
        gtype = j["gpu_type"]
        watts = float(catalog[gtype]["watts"])
        gpu_hours = float(j["hours_per_day"]) * float(j["days"]) * float(j["num_gpus"])
        kwh = gpu_energy_kwh(gpu_hours, watts)

        c_from = carbon_g(kwh * 1000.0, from_region)
        c_to = carbon_g(kwh * 1000.0, to_region)
        e_from = energy_cost_usd(kwh * 1000.0, from_region)
        e_to = energy_cost_usd(kwh * 1000.0, to_region)

        kwh_total += kwh
        carbon_from += c_from
        carbon_to += c_to
        cost_from += e_from
        cost_to += e_to
        moved += 1
        details.append({
            "job_id": str(j.get("job_id", "")), "gpu_type": gtype,
            "gpu_hours": round(gpu_hours, 1), "kwh": round(kwh, 1),
            "carbon_from_g": round(c_from, 1), "carbon_to_g": round(c_to, 1),
            "carbon_saved_g": round(c_from - c_to, 1),
            "energy_cost_from_usd": round(e_from, 2), "energy_cost_to_usd": round(e_to, 2),
        })

    saved_g = carbon_from - carbon_to
    return {
        "from_region": from_region, "to_region": to_region, "jobs_moved": moved,
        "kwh": round(kwh_total, 1),
        "carbon_from_kg": round(carbon_from / 1000.0, 2),
        "carbon_to_kg": round(carbon_to / 1000.0, 2),
        "carbon_saved_kg": round(saved_g / 1000.0, 2),
        "carbon_saved_pct": round(saved_g / carbon_from * 100.0, 1) if carbon_from > 0 else 0.0,
        "energy_cost_from_usd": round(cost_from, 2),
        "energy_cost_to_usd": round(cost_to, 2),
        "energy_cost_saved_usd": round(cost_from - cost_to, 2),
        "details": details,
    }


def reasoning_budget(rows, tokens_of=None, from_region: str = DEFAULT_REGION) -> dict:
    """Energy/carbon split for reasoning vs ordinary traffic, plus what a cap would save.

    `rows` are token_usage rows; `tokens_of` maps a row to its billable tokens.
    The 80x reasoning multiplier lives in `wh_per_query`, so this returns the
    share of energy (and CO2e) a handful of slow thinking queries really own.
    """
    def _tok(r):
        return int(tokens_of(r)) if tokens_of else int(r.get("input_tokens", 0)) + int(r.get("output_tokens", 0))

    reasoning_wh = ordinary_wh = 0.0
    reasoning_tok = ordinary_tok = 0
    n_reasoning = n_ordinary = 0
    for r in rows:
        tok = _tok(r)
        if int(float(r.get("is_reasoning", 0) or 0)):
            reasoning_wh += wh_per_query(tok, is_reasoning=True)
            reasoning_tok += tok
            n_reasoning += 1
        else:
            ordinary_wh += wh_per_query(tok)
            ordinary_tok += tok
            n_ordinary += 1

    total_wh = reasoning_wh + ordinary_wh
    return {
        "requests_reasoning": n_reasoning, "requests_ordinary": n_ordinary,
        "tokens_reasoning": reasoning_tok, "tokens_ordinary": ordinary_tok,
        "wh_reasoning": round(reasoning_wh, 2), "wh_ordinary": round(ordinary_wh, 2),
        "wh_total": round(total_wh, 2),
        "energy_share_reasoning": round(reasoning_wh / total_wh, 4) if total_wh > 0 else 0.0,
        "traffic_share_reasoning": round(n_reasoning / (n_reasoning + n_ordinary), 4) if (n_reasoning + n_ordinary) else 0.0,
        "wh_per_reasoning_query": round(reasoning_wh / n_reasoning, 2) if n_reasoning else 0.0,
        "wh_per_ordinary_query": round(ordinary_wh / n_ordinary, 2) if n_ordinary else 0.0,
        "carbon_reasoning_g": round(carbon_g(reasoning_wh, from_region), 2),
        "carbon_total_g": round(carbon_g(total_wh, from_region), 2),
        "region": from_region,
    }