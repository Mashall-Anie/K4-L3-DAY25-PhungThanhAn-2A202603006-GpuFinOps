"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}

DAYS = 30
# Extension 4 routing rule: only traffic that genuinely needs the big model gets
# the long "thinking" budget; the rest is capped at the ordinary answer length.
REASONING_ALLOWED_TIERS = ("large",)  

def _cost(row, prices=None, out_tokens=None) -> float:
    prices = prices or MODEL_PRICES
    pin, pout = prices[row["route_tier"]]
    return pricing.request_cost(
        int(num(row["input_tokens"])),
        int(num(row["output_tokens"])) if out_tokens is None else int(out_tokens),
        pin, pout,
        cached_in=int(num(row["cached_input_tokens"])),
        batch=bool(int(num(row["is_batch"]))),
    )


def cache_audit(rows) -> dict:
    """EXTENSION 3 — is the cache discount real once you pay for the writes?

    Reuse is measured from the log itself. A "session" is one hour of traffic for
    one (team, project, model): the first request in the session has to write the
    prefix, every later request in it only reads. Reads / writes is the observed
    number of sends per cached prefix — compare it with cache_break_even_reads().
    """
    sessions = defaultdict(list)
    for r in rows:
        sessions[(r["ts"][:13], r["team"], r["project"], r["model"])].append(r)
    if not sessions:
        return {"sessions": 0}

    reads = 0
    writes = 0
    dead_sessions = 0          # sessions with a single send: cache can never pay
    dead_reads = 0
    for reqs in sessions.values():
        reqs = sorted(reqs, key=lambda r: r["ts"])
        sess_reads = sum(int(num(r["cached_input_tokens"])) for r in reqs)
        writes += int(num(reqs[0]["cached_input_tokens"]))    # one prefix written per session
        reads += sess_reads
        if len(reqs) < 2 and sess_reads > 0:
            dead_sessions += 1
            dead_reads += sess_reads

    reuse = reads / writes if writes else 0.0
    verdict = pricing.cache_is_worth_it(reuse)

    # price the haircut on the *model mix actually used* in this traffic
    gross = 0.0
    for r in rows:
        cached = int(num(r["cached_input_tokens"]))
        if cached <= 0:
            continue
        pin, _ = MODEL_PRICES[r["route_tier"]]
        batch_mult = 0.50 if int(num(r["is_batch"])) else 1.0
        billed = cached / 1e6 * pin * batch_mult
        gross += billed * (1.0 - pricing.CACHE_READ_DISCOUNT)   # the "90% off" claim
    return {
        "sessions": len(sessions),
        "reads_tokens": reads,
        "write_proxy_tokens": writes,
        "reuse": round(reuse, 3),
        "verdict": verdict,
        "dead_sessions": dead_sessions,
        "dead_reads_tokens": dead_reads,
        "gross_daily_savings": round(gross, 4),
        "audited_daily_savings": round(gross * verdict["net_saving_frac"], 4),
        "audited_monthly_savings": round(gross * verdict["net_saving_frac"] * DAYS, 2),
        "haircut_daily_usd": round(gross * verdict["discount_haircut"], 4),
        "not_worth_sessions_pct": round(dead_sessions / len(sessions) * 100.0, 1),
    }


def reasoning_audit(rows) -> dict:
    """EXTENSION 4 — what the reasoning traffic costs in $ and in Wh, and a cap rule."""
    reasoning = [r for r in rows if int(num(r["is_reasoning"])) == 1]
    ordinary = [r for r in rows if int(num(r["is_reasoning"])) == 0]
    if not reasoning or not ordinary:
        return {"requests_reasoning": 0}

    cost_reasoning = sum(_cost(r) for r in reasoning)
    cost_all = sum(_cost(r) for r in rows)

    energy = sustainability.reasoning_budget(rows)
    # Cap the thinking budget for traffic that is not on the big model: keep the
    # ordinary answer length observed on the same tier instead of a 4k-token essay.
    cap = sorted(int(num(r["output_tokens"])) for r in ordinary)[len(ordinary) // 2]
    capped = 0
    saved_cost = 0.0
    saved_tokens = 0
    saved_wh = 0.0
    for r in reasoning:
        if r["route_tier"] in REASONING_ALLOWED_TIERS:
            continue
        out = int(num(r["output_tokens"]))
        capped_out = min(out, cap)
        saved_cost += _cost(r) - _cost(r, out_tokens=capped_out)
        saved_tokens += out - capped_out
        tok_in = int(num(r["input_tokens"]))
        saved_wh += (sustainability.wh_per_query(tok_in + out, is_reasoning=True)
                     - sustainability.wh_per_query(tok_in + capped_out, is_reasoning=True))
        capped += 1

    return {
        "requests_reasoning": len(reasoning),
        "requests_ordinary": len(ordinary),
        "traffic_share": round(len(reasoning) / len(rows), 4),
        "token_share_output": round(sum(int(num(r["output_tokens"])) for r in reasoning)
                                    / sum(int(num(r["output_tokens"])) for r in rows), 4),
        "cost_reasoning_daily": round(cost_reasoning, 4),
        "cost_all_daily": round(cost_all, 4),
        "cost_share": round(cost_reasoning / cost_all, 4) if cost_all else 0.0,
        "cap_tokens": cap,
        "requests_capped": capped,
        "capped_daily_savings": round(saved_cost, 4),
        "capped_monthly_savings": round(saved_cost * DAYS, 2),
        "capped_tokens": saved_tokens,
        "capped_wh_daily": round(saved_wh, 2),
        "capped_wh_monthly": round(saved_wh * DAYS, 2),
        "energy": energy,
    }


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        opt_cost += pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    cache = cache_audit(rows)
    reasoning = reasoning_audit(rows)

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        v = cache["verdict"]
        print(f"\n-- Extension 3: is the cache discount real? --")
        print(f"sessions={cache['sessions']}  reads={cache['reads_tokens']:,} tokens  "
              f"writes~{cache['write_proxy_tokens']:,} tokens  -> reuse {cache['reuse']:.2f}x")
        print(f"break-even {v['break_even_reads']:.2f} sends per prefix  ->  "
              f"worth_it={v['worth_it']} (margin {v['margin']:.2f}x)")
        print(f"advertised 90% off  ->  {v['net_saving_frac']:.1%} off after the 1.25x write "
              f"(haircut {v['discount_haircut']:.1%})")
        print(f"audited cache saving: ${cache['audited_daily_savings']:.2f}/day "
              f"(${cache['audited_daily_savings'] * DAYS:,.2f}/month); "
              f"{cache['dead_sessions']} sessions ({cache['not_worth_sessions_pct']}%) never re-read -> cache OFF there")

        e = reasoning["energy"]
        print(f"\n-- Extension 4: reasoning budget --")
        print(f"reasoning = {reasoning['traffic_share']:.1%} of requests but "
              f"{reasoning['token_share_output']:.1%} of output tokens, "
              f"{reasoning['cost_share']:.1%} of the optimized bill")
        print(f"energy: {e['wh_reasoning'] / 1000:.1f} kWh ({e['energy_share_reasoning']:.1%}) of "
              f"{e['wh_total'] / 1000:.1f} kWh  ->  {e['wh_per_reasoning_query']:.0f} Wh/query vs "
              f"{e['wh_per_ordinary_query']:.2f} Wh/query")
        print(f"cap rule: reasoning only on {REASONING_ALLOWED_TIERS}, others capped at "
              f"{reasoning['cap_tokens']} output tokens")
        print(f"  -> {reasoning['requests_capped']:.0f} requests capped, "
              f"${reasoning['capped_monthly_savings']:,.2f}/month saved, "
              f"{reasoning['capped_wh_monthly'] / 1000:,.1f} kWh/month avoided")


    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache": cache, "reasoning": reasoning,

    }


if __name__ == "__main__":
    run()
