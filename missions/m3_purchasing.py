"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py

Adds two "Your Turn" extensions:
  * EXTENSION 1 — risk-adjusted tier policy (finops/pricing.tier_decision): prices
    every tier on the job's own hours, uses per-GPU-type interruption rates, and
    refuses reserved commitments the workload cannot amortise.
  * EXTENSION 5 — carbon-aware scheduling: moves interruptible jobs to the
    cleanest/cheapest region and prices the CO2e and electricity saved.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing, sustainability

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = baseline_monthly = 0.0
    recs = []
    phantom_total = 0.0
    rework_hours_total = 0.0
    rework_cost_total = 0.0

    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        # A 24/7 inference service is a persistent workload (no end date); a batch
        # job ends when `days` runs out, and that decides which commitments fit.
        persistent = j["kind"] == "infer" and int(num(j["days"])) >= 28
        decision = pricing.tier_decision(
            hours_per_day=hpd, interruptible=interruptible, gpu_type=gtype,
            job_days=None if persistent else int(num(j["days"])),
            on_demand_hr=od, spot_hr=num(c["spot_hr"]),
            reserved_1yr_hr=num(c["reserved_1yr_hr"]), reserved_3yr_hr=num(c["reserved_3yr_hr"]),
        )
        baseline_tier = pricing.recommend_tier(hpd, interruptible)   # the simple policy
        baseline_cost = {
            "spot": pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)["spot_cost"],
            "reserved": gpu_hours * num(c["reserved_3yr_hr"]),
            "on_demand": on_demand_cost,
        }[baseline_tier]

        opt_cost = decision["cost"] * ngpu              # per-GPU month -> whole job
        tier = decision["tier"]

        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od,
                                               interrupt_rate=pricing.interrupt_rate_for(gtype))
            rework_hours_total += sim["spot_effective_hours"] - gpu_hours
            rework_cost_total += sim["spot_cost"] - gpu_hours * num(c["spot_hr"])

        on_demand_monthly += on_demand_cost
        baseline_monthly += baseline_cost
        optimized_monthly += opt_cost
        phantom_total += decision["phantom_savings"] * ngpu
        recs.append({
            "job_id": j["job_id"], "gpu_type": gtype, "tier": tier, "term": decision["term"],
            "baseline_tier": baseline_tier,
            "on_demand": round(on_demand_cost), "optimized": round(opt_cost),
            "baseline_policy": round(baseline_cost),
            "interrupt_rate": pricing.interrupt_rate_for(gtype),
            "phantom_savings": round(decision["phantom_savings"] * ngpu, 2),
            "reason": decision["reason"],
            "spot_margin": decision["spot_interrupt_margin"],
        })

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0
    tiers = sorted({r["tier"] for r in recs})

    # --- EXTENSION 5: carbon-aware scheduling for the interruptible jobs ---
    carbon = sustainability.carbon_aware_plan(jobs, cat)
    # price of flexibility: what a 1yr commitment would cost instead of a 3yr one
    term_1yr_monthly = 0.0
    for j in jobs:
        if (j["kind"] == "infer" and int(num(j["days"])) >= 28):
            term_1yr_monthly += num(j["hours_per_day"]) * DAYS * int(num(j["num_gpus"])) * num(cat[j["gpu_type"]]["reserved_1yr_hr"])

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':6}{'tier':9}{'term':5}{'int%':>6}{'on-demand':>11}{'simple':>10}{'optimized':>11}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:6}{r['tier']:9}{r['term']:5}"
                  f"{r['interrupt_rate']:>6.0%}${r['on_demand']:>10,}${r['baseline_policy']:>9,}${r['optimized']:>10,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")
        print(f"tiers used: {tiers}")

        print(f"\n-- Extension 1: risk-adjusted tier policy --")
        print(f"simple policy   : ${baseline_monthly:,.0f}/month")
        print(f"risk-adjusted   : ${optimized_monthly:,.0f}/month  "
              f"(delta ${baseline_monthly - optimized_monthly:+,.0f})")
        print(f"phantom savings refused (commitments no job could amortise): ${phantom_total:,.0f}/month")
        print(f"spot rework paid: {rework_hours_total:,.0f} GPU-hours (${rework_cost_total:,.0f}) "
              f"-> the real price of interruption risk")
        print(f"1yr instead of 3yr on the 3 persistent services would cost "
              f"${term_1yr_monthly - sum(r['optimized'] for r in recs if r['term'] == '3yr'):,.0f}/month more "
              f"(the price of flexibility)")

        print(f"\n-- Extension 5: carbon-aware scheduling --")
        print(f"move {carbon['jobs_moved']} interruptible jobs: {carbon['from_region']} -> {carbon['to_region']}")
        print(f"energy {carbon['kwh']:,.0f} kWh: {carbon['carbon_from_kg']:,.0f} -> "
              f"{carbon['carbon_to_kg']:,.0f} kgCO2e ({carbon['carbon_saved_pct']:.0f}% less), "
              f"electricity ${carbon['energy_cost_from_usd']:,.0f} -> ${carbon['energy_cost_to_usd']:,.0f} "
              f"(saves ${carbon['energy_cost_saved_usd']:,.0f})")

    return {
        "recommendations": recs,
        "on_demand_monthly": round(on_demand_monthly),
        "baseline_policy_monthly": round(baseline_monthly),
        "optimized_monthly": round(optimized_monthly),
        "savings_pct": round(savings_pct, 1),
        "savings_usd": round(savings, 2),
        "tiers": tiers,
        "phantom_savings_usd": round(phantom_total, 2),
        "term_1yr_monthly": round(term_1yr_monthly, 2),
        "spot_rework_hours": round(rework_hours_total, 1),
        "spot_rework_cost_usd": round(rework_cost_total, 2),
        "carbon": carbon,
    }


if __name__ == "__main__":
    run()
