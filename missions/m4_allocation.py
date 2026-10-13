"""M4 — Cost Allocation: tag -> showback -> chargeback + FOCUS export (deck §10).

Run: python missions/m4_allocation.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import csv
import os
from missions._common import load_csv, num, ROOT
from finops import allocation
from missions.m2_inference_levers import MODEL_PRICES
from finops import pricing

TAGS = ["team", "project"]
FOCUS_COLUMNS = ["BillingAccountId", "ChargePeriodStart", "ChargePeriodEnd", "ServiceCategory",
                 "ServiceName", "ResourceId", "ResourceType", "SkuId", "BilledCost",
                 "BillingCurrency", "team", "project"]

def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    priced = []
    for i, r in enumerate(rows, start=1):
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        pin, pout = MODEL_PRICES[r["route_tier"]]
        cost = pricing.request_cost(inp, out, pin, pout,
                                    cached_in=int(num(r["cached_input_tokens"])),
                                    batch=bool(int(num(r["is_batch"]))))
        priced.append({
            "team": r["team"], "project": r["project"], "service": "gpu-inference",
            "resource_id": f"llm-req-{i:05d}", "resource_type": "LLM request",
            "sku": f"{r['route_tier']}-model", "period": r["ts"][:10],
            "cost": cost, "tokens": inp + out,
        })

    by_team = allocation.cost_by_tag(priced, "team")
    by_project = allocation.cost_by_tag(priced, "project")
    coverage = allocation.tag_coverage(priced, TAGS)
    ready = allocation.chargeback_ready(coverage)

    # per-team token efficiency: the number that actually drives the bill
    team_tokens = allocation.cost_by_tag(priced, "team", cost_key="tokens")
    team_unit = {t: round(by_team[t] / (team_tokens[t] / 1e6), 3) for t in by_team if team_tokens.get(t)}
    untagged = sum(v for k, v in by_team.items() if k == "(untagged)")

    # write the FOCUS-style export — every request, not a sample
    focus = allocation.to_focus_rows(priced)

    out_path = os.path.join(ROOT, "outputs", "focus_export.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FOCUS_COLUMNS)
        for fr, pr in zip(focus, priced):
            w.writerow([fr["BillingAccountId"], fr["ChargePeriodStart"], pr["period"],
                        fr["ServiceCategory"], fr["ServiceName"], fr["ResourceId"],
                        pr["resource_type"], pr["sku"], fr["BilledCost"],
                        fr["BillingCurrency"], fr["Tags"]["team"], fr["Tags"]["project"]])

    if verbose:
        print("== M4 Cost Allocation ==")
        print("cost by team ($/day):")
        for k, v in sorted(by_team.items(), key=lambda x: -x[1]):
            print(f"  {k:12} ${v:8.2f}   {team_unit.get(k, 0):6.3f} $/1M-token")
        print("cost by project ($/day):")
        for k, v in sorted(by_project.items(), key=lambda x: -x[1]):
            print(f"  {str(k):14} ${v:8.2f}")
        print(f"tag coverage: {coverage:.0%}  ->  chargeback ready? {ready}")
        if untagged:
            print(f"untagged spend: ${untagged:.2f}/day (inside the team numbers above)")
        print(f"FOCUS export -> outputs/focus_export.csv ({len(focus)} rows, "
            f"${sum(p['cost'] for p in priced):.2f}/day)")
        
    return {"by_team": {k: round(v, 2) for k, v in by_team.items()},
            "by_project": {k: round(v, 2) for k, v in by_project.items()},
            "team_unit_economics": team_unit,
            "team_tokens": {k: int(v) for k, v in team_tokens.items()},
            "tag_coverage": round(coverage, 3), "chargeback_ready": ready,
            "focus_rows": len(focus), "untagged_daily": round(untagged, 2),
            "total_daily": round(sum(p["cost"] for p in priced), 2)}

if __name__ == "__main__":
    run()
