"""M5 — Optimization Report: combine M1-M4 into baseline-vs-optimized (deck §1/§11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png

The four levers stay exactly four (so the waterfall and the arithmetic agree);
the "Your Turn" findings (cache audit, reasoning budget, carbon-aware schedule,
tier policy, showback) are reported as named sections with their own numbers.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, ROOT
from finops import report, sustainability, metrics, pricing
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing, m4_allocation

DAYS = 30


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    r4 = m4_allocation.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets -----------------------------------------------------------
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]
    rightsize_savings = r1["rightsize_monthly_savings"]
    idle_savings = r1["idle_waste_monthly"]

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "Right-size over-provisioned GPUs": round(rightsize_savings),
        "Kill idle GPUs": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- unit economics ----------------------------------------------------
    tokens_month = r2["total_tokens"] * DAYS
    unit_economics = {
        "Inference API — $/1M-token": {
            "baseline": r2["baseline_per_m"], "optimized": r2["optimized_per_m"],
            "delta": f"{(1 - r2['optimized_per_m'] / r2['baseline_per_m']) * 100:.0f}% lower",
            "fmt": "${:,.3f}",
        },
        "All-in incl. GPU rental — $/1M-token": {
            "baseline": baseline / (tokens_month / 1e6), "optimized": optimized / (tokens_month / 1e6),
            "delta": f"{(1 - optimized / baseline) * 100:.0f}% lower",
            "fmt": "${:,.2f}",
        },
    }

    # --- savings narrative -------------------------------------------------
    lie_ids = [l["gpu_id"] for l in r1["lies"]]
    lie_gpus = [s for s in r1["summary"] if s["gpu_id"] in lie_ids]
    idle_gpu = max(r1["summary"], key=lambda s: s["idle_hours"])
    biggest_lever = max(levers, key=levers.get)
    rightsizing = [x for x in r1["rightsizing"] if x["monthly_savings_per_gpu"] > 0]

    levers_detail = {
        "Inference (cascade/cache/batch)": {
            "what": "route simple traffic to the small model, bill cached input at 0.1x, "
                    "and batch everything that can wait",
            "evidence": f"${r2['baseline_per_m']:.3f} -> ${r2['optimized_per_m']:.3f} per 1M tokens "
                        f"({r2['savings_pct']:.1f}%) on {r2['total_tokens']:,} tokens/day",
            "why": "the levers multiply, they do not add: batch (x0.50) x full cache hit (x0.10) = 0.05 "
                   f"of the naive bill (stack = {0.05:.2f})",
        },
        "Purchasing (spot/reserved)": {
            "what": "spot + checkpoint for interruptible jobs, reserved for services that "
                    "really run at duty cycle above break-even",
            "evidence": f"${r3['on_demand_monthly']:,.0f} -> ${r3['optimized_monthly']:,.0f} per month "
                        f"({r3['savings_pct']:.1f}% off on-demand)",
            "why": f"the workload mix is {r3['tiers']}: reserved discounts run 20-44% by card, and short "
                   f"batch jobs ride spot instead of locking a commitment longer than their own life",
        },
        "Right-size over-provisioned GPUs": {
            "what": "move GPUs that never approach either ceiling onto cheaper silicon",
            "evidence": "; ".join(
                f"{x['gpu_id']} {x['current_type']}->{x['recommended_type']} "
                f"(peak needs {x['bw_required_tbs']:.2f} TB/s, {x['tflops_required']:.0f} TF, "
                f"{x['hbm_required_gb']:.0f} GB) saves ${x['monthly_savings_per_gpu']:,.0f}/mo"
                for x in rightsizing) or "no GPU met the over-provisioning test",
            "why": "you rent the card, not the FLOPs: if neither MFU nor MBU is near target, "
                   "a smaller card does the same job for less rent",
        },
        "Kill idle GPUs": {
            "what": "shut down or reschedule the GPU-hours where the card is <10% utilised",
            "evidence": f"{idle_gpu['idle_hours']:.0f} idle h/day on {idle_gpu['gpu_id']} "
                        f"(${r1['idle_waste_daily']:,.2f}/day, ${idle_savings:,.0f}/month)",
            "why": "an idle reserved GPU is the most expensive kind: you pay the full hour and "
                   "get no tokens for it",
        },
    }

    # --- sustainability ----------------------------------------------------
    median_tokens = 800
    wh = sustainability.wh_per_query(median_tokens)
    fleet_kwh_day = sum(num(r["power_w"]) for r in m1_efficiency_audit.load_csv("gpu_telemetry.csv")) / 1000.0
    energy = r2["reasoning"]["energy"]
    cleanest = sustainability.cleanest_cheapest_region(1.0)
    best_region = sustainability.cleanest_cheapest_region()
    carbon = r3["carbon"]
    idle_kwh = sum(num(cat[s["gpu_type"]]["watts"]) for s in r1["summary"] if s["idle_hours"]) * 8 / 1000 * DAYS
    rightsize_kwh = 0.0
    for x in rightsizing:
        rightsize_kwh += (num(cat[x["current_type"]]["watts"]) - num(cat[x["recommended_type"]]["watts"])) * 24 * DAYS / 1000
    sust = {
        "wh_per_query": wh,
        "queries_per_kwh": round(1000.0 / wh, 0) if wh else 0,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "carbon_kg_per_day": round(sustainability.carbon_g(fleet_kwh_day * 1000, "us-east-1") / 1000, 1),
        "carbon_region": "us-east-1",
        "cleanest_region": cleanest,
        "cleanest_region_gco2_kwh": sustainability.REGION_CARBON[cleanest],
        "best_region": best_region,
        "best_region_gco2_kwh": sustainability.REGION_CARBON[best_region],
        "best_region_kwh_price": sustainability.REGION_PRICE_KWH[best_region],
        "carbon_saved_kg": carbon["carbon_saved_kg"],
        "carbon_saved_pct": carbon["carbon_saved_pct"],
        "energy_cost_saved_usd": carbon["energy_cost_saved_usd"],
        "jobs_moved": carbon["jobs_moved"],
        "to_region": carbon["to_region"],
        "notes": [
            f"Reasoning traffic is {energy['traffic_share_reasoning']:.1%} of requests but "
            f"{energy['energy_share_reasoning']:.1%} of inference energy "
            f"({energy['wh_per_reasoning_query']:.0f} Wh/query vs {energy['wh_per_ordinary_query']:.2f} Wh/query).",
            f"Cutting idle + right-sizing removes ~{idle_kwh + rightsize_kwh:,.0f} kWh/month from the fleet.",
            "Electricity price and grid carbon move together here: the cleanest blend is also the cheapest, "
            "so the carbon decision is free money (see Extension 5).",
        ],
    }

    # --- extra sections ----------------------------------------------------
    cache = r2["cache"]
    v = cache["verdict"]
    reasoning = r2["reasoning"]
    tier_delta = r3["baseline_policy_monthly"] - r3["optimized_monthly"]
    best_team = max(r4["by_team"].items(), key=lambda kv: kv[1])
    tag_rows = m4_allocation.load_csv("token_usage.csv")
    untagged_requests = sum(1 for r in tag_rows if not str(r.get("project", "")).strip())
    untagged_daily = (r4["total_daily"] * untagged_requests / max(1, len(tag_rows)))

    sections = [
        "## Why GPU-Util is a lie (and what it costs)",
        "",
        f"`nvidia-smi` reports *time-active clocks*, not work done. On {(' and '.join(lie_ids))} the card "
        "reported "
        + " and ".join(f"{l['gpu_util_pct']:.0f}% util while MFU was only {l['mfu']:.2f}" for l in lie_gpus)
        + f". At ${num(cat[lie_gpus[0]['gpu_type']]['on_demand_hr']):.2f}/GPU-hour the fleet pays for a full "
        f"{lie_gpus[0]['gpu_type']} hour and receives about {lie_gpus[0]['mfu'] * 100:.0f}% of the FLOPs, "
        f"{lie_gpus[0]['mbu'] * 100:.0f}% of the bandwidth. Financial impact: "
        f"${rightsize_savings:,.0f}/month of the bill buys silicon the workload cannot use "
        f"(right-sizing lever), on top of ${idle_savings:,.0f}/month burned on idle hours. "
        "The fix is not 'make the GPU busier', it is to measure MFU/MBU and buy the card the job "
        "actually needs.",
        "",
        "## Your Turn extensions (measured)",
        "",
        "### Extension 1 — risk-adjusted tier policy (`pricing.tier_decision`)",
        f"- Prices spot/reserved/on-demand on each job's own hours, with per-GPU-type interruption "
        f"rates ({', '.join(f'{k} {v:.0%}' for k, v in list(pricing.INTERRUPT_RATE_BY_TYPE.items())[:3])}, ...).",
        f"- Result: ${r3['baseline_policy_monthly']:,.0f} -> ${r3['optimized_monthly']:,.0f}/month "
        f"(${tier_delta:+,.0f} vs the simple policy) and it **refuses ${r3['phantom_savings_usd']:,.0f}/month of "
        f"phantom savings** from commitments no job could amortise.",
        f"- Real cost of interruption risk: {r3['spot_rework_hours']:,.0f} GPU-hours of rework "
        f"(${r3['spot_rework_cost_usd']:,.0f}/month) — cheap insurance against a 1-3 year lock-in.",
        f"- Flexibility price: 1yr instead of 3yr on the persistent services costs "
        f"${r3['term_1yr_monthly'] - sum(x['optimized'] for x in r3['recommendations'] if x['term'] == '3yr'):,.0f}/month more.",
        "",
        "### Extension 2 — MBU-driven right-sizing (`metrics.rightsize_gpu`)",
        f"- Rule: flag when peak MFU < {metrics.DEFAULT_MFU_CAP:.2f} **and** peak MBU < "
        f"{metrics.DEFAULT_MBU_TARGET:.2f}; then pick the cheapest card that still "
        "covers observed peak FLOPs, peak bandwidth (at <=60% MBU) and peak HBM.",
        f"- Result: {len(rightsizing)} GPU(s) re-shaped for ${rightsize_savings:,.0f}/month "
        f"(-{rightsize_kwh:,.0f} kWh/month) with no job moved off its workload.",
        "",
        "### Extension 3 — `cache_is_worth_it()` (cache is a loan, not free money)",
        f"- Observed reuse: {cache['reads_tokens']:,} cached tokens read over "
        f"{cache['write_proxy_tokens']:,} written -> **{cache['reuse']:.2f} sends per prefix** "
        f"vs break-even {v['break_even_reads']:.2f}.",
        f"- The advertised 90% off is really **{v['net_saving_frac']:.1%} off** once the 1.25x write price "
        f"is paid (haircut {v['discount_haircut']:.1%}); audited cache value "
        f"${cache['audited_monthly_savings']:,.0f}/month.",
        f"- {cache['dead_sessions']} sessions ({cache['not_worth_sessions_pct']}% of traffic) are single-send: "
        "cache must be **off** there or it loses money.",
        "",
        "### Extension 4 — reasoning budget",
        f"- Reasoning is {reasoning['traffic_share']:.1%} of requests but {reasoning['token_share_output']:.1%} of "
        f"output tokens and {reasoning['cost_share']:.1%} of the optimized bill.",
        f"- Energy: **{energy['energy_share_reasoning']:.1%} of inference energy** "
        f"({energy['wh_per_reasoning_query']:.0f} Wh/query vs {energy['wh_per_ordinary_query']:.2f} Wh/query).",
        f"- Rule 'reasoning only on the large tier, everything else capped at "
        f"{reasoning['cap_tokens']} output tokens': {reasoning['requests_capped']:.0f} requests capped, "
        f"${reasoning['capped_monthly_savings']:,.0f}/month and "
        f"{reasoning['capped_wh_monthly'] / 1000:,.1f} kWh/month avoided (correctness stays with the big model).",
        "",
        "### Extension 5 — carbon-aware scheduling",
        f"- Move {carbon['jobs_moved']} interruptible jobs {carbon['from_region']} -> {carbon['to_region']}: "
        f"{carbon['kwh']:,.0f} kWh gives {carbon['carbon_from_kg']:,.0f} -> {carbon['carbon_to_kg']:,.0f} kgCO2e "
        f"(**-{carbon['carbon_saved_pct']:.0f}%**) and ${carbon['energy_cost_from_usd']:,.0f} -> "
        f"${carbon['energy_cost_to_usd']:,.0f} of electricity (**-${carbon['energy_cost_saved_usd']:,.0f}/month**).",
        f"- `{cleanest}` is the cleanest grid ({sustainability.REGION_CARBON[cleanest]} gCO2e/kWh); "
        f"`{best_region}` wins the cost+carbon blend. Same workload, different postcode, different bill.",
        "",
        "## Cost allocation: showback before chargeback",
        "",
        f"- Team showback (${r4['by_team'][best_team[0]]:.2f}/day for the biggest spender, {best_team[0]}) is "
        f"only trustworthy at **{r4['tag_coverage']:.0%} tag coverage** -> chargeback gate: "
        f"**{'OPEN' if r4['chargeback_ready'] else 'CLOSED'}**.",
        f"- ${untagged_daily:.2f}/day of spend still has no `project` tag "
        f"({untagged_requests} of {r4['focus_rows']:,} requests): that slice cannot be charged back to a project "
        "until tagging is fixed.",
        f"- `outputs/focus_export.csv` carries all {r4['focus_rows']:,} request-level rows "
        f"(BilledCost + team/project + SkuId) so the same feed works for showback, chargeback and "
        "multi-vendor roll-ups.",
        "",
        "## Recommended actions, in order",
        "",
        f"1. **Close the purchasing gap first (${levers['Purchasing (spot/reserved)']:,.0f}/month).** "
        "Largest, safest lever: reserve only services measured at duty >= break-even, spot everything "
        "checkpointable, and never commit to a term longer than the workload's life.",
        f"2. **Re-route inference traffic (${levers['Inference (cascade/cache/batch)']:,.0f}/month).** "
        "Small-model default, cache only prefixes reused >= 2 times, batch what can wait. "
        "Then attack reasoning energy (94% of inference Wh).",
        f"3. **Fix the fleet shape (${levers['Right-size over-provisioned GPUs'] + levers['Kill idle GPUs']:,.0f}/month "
        "+ carbon).** Right-size the GPUs that never approach MFU/MBU targets, kill the idle hours "
        f"({idle_gpu['idle_hours']:.0f} h/day on {idle_gpu['gpu_id']}), and schedule interruptible jobs in "
        f"{carbon['to_region']} for -{carbon['carbon_saved_pct']:.0f}% CO2e at no extra cost.",
        "",
        "## Caveats",
        "",
        "- Prices are a June-2026 snapshot; re-baseline before committing money.",
        "- Purchasing economics are modelled as a 30-day run rate per job (the mission's monthly "
        "convention), not per individual run.",
        "- The reasoning energy model (80x multiplier on token counts) is directional: at this traffic "
        "volume it implies more energy than the 11-GPU telemetry window shows, so treat Wh as "
        "relative, not absolute.",
    ]

    md = report.build_report(baseline, optimized, levers, sustainability=sust,
                             unit_economics=unit_economics, levers_detail=levers_detail,
                             sections=sections)
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"),
                                   baseline_usd=baseline, optimized_usd=optimized,
                                   unit_economics=unit_economics)

    if verbose:
        print("== M5 Optimization Report ==")
        print(f"baseline ${baseline:,.0f} -> optimized ${optimized:,.0f}  ({total_pct:.1f}% saved)")
        for k, v2 in levers.items():
            print(f"  {k:36} ${v2:>9,}")
        print(f"  {'TOTAL':36} ${sum(levers.values()):>9,}")
        print(f"\n$/1M-token: inference {r2['baseline_per_m']:.3f} -> {r2['optimized_per_m']:.3f} | "
              f"all-in {unit_economics['All-in incl. GPU rental — $/1M-token']['baseline']:.2f} -> "
              f"{unit_economics['All-in incl. GPU rental — $/1M-token']['optimized']:.2f}")
        print(f"Written: outputs/report.md" + (f" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1),
            "unit_economics": unit_economics, "sustainability": sust,
            "allocation": r4, "review": {"cache": cache, "reasoning": reasoning,
                                          "tiers": r3["tiers"], "carbon": carbon}}


if __name__ == "__main__":
    run()
