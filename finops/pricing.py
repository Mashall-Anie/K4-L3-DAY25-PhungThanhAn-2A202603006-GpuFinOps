"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


def recommend_tier(hours_per_day: float, interruptible: bool, reserved_discount: float = 0.45,
                   gpu_type: str | None = None, job_days: int | None = None,
                   on_demand_hr: float | None = None, spot_hr: float | None = None,
                   reserved_hr: float | None = None) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    BASELINE simple policy (kept as the fallback when no prices are supplied so
    existing callers/tests keep working):
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)

    EXTENSION 1 — risk-adjusted, cost-based policy (used when prices are given).
    See `tier_decision()` for the model: it prices all three tiers on this job's
    own hours, uses per-GPU-type interruption rates, gates reserved on the
    commitment horizon and on duty >= break-even for that specific discount, and
    returns the cheapest *eligible* option.
    """
    if on_demand_hr is None:
        duty = max(0.0, hours_per_day) / 24.0
        be = break_even_utilization(reserved_discount)
        if interruptible and hours_per_day < 24:
            return "spot"
        if duty >= be:
            return "reserved"
        return "on_demand"

    decision = tier_decision(
        hours_per_day=hours_per_day, interruptible=interruptible, gpu_type=gpu_type,
        job_days=job_days, on_demand_hr=on_demand_hr, spot_hr=spot_hr,
        reserved_hr=reserved_hr, reserved_discount=reserved_discount,
    )
    return decision["tier"]


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }


# ===========================================================================
# EXTENSION 1 — risk-adjusted purchasing policy
#
# The simple policy has three blind spots that cost real money:
#   1. It ignores *how likely* a spot GPU is to be reclaimed. An A10G dev box is
#      interrupted far more often than an H100 training run, and every
#      interruption throws away the compute since the last checkpoint.
#   2. It never asks whether a reserved commitment is even *available* for this
#      job. Reserving 3 years of H100 to serve a 3-day finetune is not a
#      discount, it is a liability. A per-hour price comparison that ignores the
#      commitment horizon produces "phantom savings" you can never bank.
#   3. It treats the reserved discount as a constant, so it never checks the duty
#      cycle against the break-even of the discount it is actually getting.
# ===========================================================================

# Per-GPU-type hourly interruption probability on the spot market (2026 snapshot).
# Big training cards in high demand are reclaimed less often than cheap dev cards.
INTERRUPT_RATE_BY_TYPE = {
    "H100": 0.03, "H200": 0.03, "B200": 0.04,
    "A100": 0.06, "MI300X": 0.05,
    "A10G": 0.12, "L4": 0.12,
}
DEFAULT_INTERRUPT_RATE = 0.05

# Commitment horizons, in days.
COMMITMENT_TERMS = {"1yr": 365, "3yr": 1095}
DEFAULT_JOB_LIFETIME_DAYS = 1095  # a 24/7 service with no end date: long enough for a 3yr term


def interrupt_rate_for(gpu_type: str | None) -> float:
    """Expected hourly spot interruption rate for a GPU type."""
    if gpu_type is None:
        return DEFAULT_INTERRUPT_RATE
    return INTERRUPT_RATE_BY_TYPE.get(gpu_type, DEFAULT_INTERRUPT_RATE)


def reserved_term_for(job_days: int | None, terms: dict | None = None) -> str:
    """Longest commitment the workload can actually amortise.

    Anything that will not still be running when the term ends cannot use the
    term's hourly rate for its whole life, so it is capped at the shorter term.
    """
    terms = terms or COMMITMENT_TERMS
    lifetime = DEFAULT_JOB_LIFETIME_DAYS if job_days is None else max(0, job_days)
    if lifetime >= terms["3yr"]:
        return "3yr"
    if lifetime >= terms["1yr"]:
        return "1yr"
    return "none"


def spot_interruption_breakeven(job_hours: float, spot_hr: float, on_demand_hr: float,
                                ckpt_overhead_frac: float = 0.03,
                                rework_hours_per_interrupt: float = 0.5) -> float:
    """Interruption rate at which spot stops being cheaper than on-demand.

    Solve for r:  (job_hours*(1+ckpt) + job_hours*r*rework) * spot_hr = job_hours * on_demand_hr
    A safety margin = breakeven_rate / expected_rate tells you how much spot
    prices/availability can degrade before the decision flips.
    """
    if job_hours <= 0 or rework_hours_per_interrupt <= 0:
        return 0.0
    budget = (on_demand_hr / spot_hr if spot_hr > 0 else 0.0) - (1.0 + ckpt_overhead_frac)
    return max(0.0, budget / rework_hours_per_interrupt)


def tier_decision(hours_per_day: float, interruptible: bool,
                  gpu_type: str | None = None, job_days: int | None = None,
                  on_demand_hr: float | None = None, spot_hr: float | None = None,
                  reserved_hr: float | None = None, reserved_1yr_hr: float | None = None,
                  reserved_3yr_hr: float | None = None, reserved_discount: float = 0.45,
                  days_per_month: int = 30) -> dict:
    """EXTENSION 1 — price every tier on this job's own hours and pick the cheapest
    *eligible* one; report the options it had to reject and why.

    Returns a dict with the chosen tier/term, the monthly cost, the rejection
    reasons, and `phantom_savings` = money a naive "cheapest hourly rate" policy
    would have claimed by committing to a term the job can never amortise.
    """
    if on_demand_hr is None or on_demand_hr <= 0:
        raise ValueError("on_demand_hr is required for tier_decision()")

    hours = max(0.0, hours_per_day) * days_per_month          # per-GPU GPU-hours billed in a month
    duty = max(0.0, hours_per_day) / 24.0
    od_rate = float(on_demand_hr)
    spot_rate = float(spot_hr) if spot_hr else od_rate * 0.60
    res_discount = reserved_discount if reserved_hr is None else (1.0 - float(reserved_hr) / od_rate)
    res_rate = float(reserved_hr) if reserved_hr is not None else od_rate * (1.0 - reserved_discount)

    term = reserved_term_for(job_days)
    lifetime = DEFAULT_JOB_LIFETIME_DAYS if job_days is None else max(0, job_days)

    rate = interrupt_rate_for(gpu_type)
    sim = spot_checkpoint_cost(hours, spot_rate, od_rate, interrupt_rate=rate)
    spot_cost = sim["spot_cost"]

    # One reserved option per commitment horizon the workload can amortise.
    reserved_terms = []
    if reserved_1yr_hr or reserved_3yr_hr:
        for label, rhr in (("3yr", reserved_3yr_hr), ("1yr", reserved_1yr_hr)):
            if rhr is None:
                continue
            if reserved_term_for(job_days, {"1yr": COMMITMENT_TERMS["1yr"] if label == "1yr" else 10**9,
                                            "3yr": COMMITMENT_TERMS["3yr"] if label == "3yr" else 10**9}) != label:
                continue
            reserved_terms.append((label, float(rhr),
                                   break_even_utilization(1.0 - float(rhr) / od_rate)))
        if not reserved_terms:
            term = "none"
    else:
        reserved_terms.append((term if term != "none" else "3yr", res_rate,
                               break_even_utilization(res_discount)))
        if term == "none":
            reserved_terms = []

    be = break_even_utilization(res_discount)
    reserved_ok = bool(reserved_terms) and (duty >= min(t[2] for t in reserved_terms))

    options = [
        {
            "tier": "spot", "term": "none", "cost": round(spot_cost, 2),
            "eligible": bool(interruptible),
            "reason": ("job is checkpointable, spot discount survives rework"
                       if interruptible else "workload is not interruptible"),
            "effective_hours": sim["spot_effective_hours"],
            "interrupt_rate": rate,
        },
    ]
    if reserved_terms:
        for label, rhr, be_lbl in reserved_terms:
            options.append({
                "tier": "reserved", "term": label, "cost": round(hours * rhr, 2),
                "eligible": bool(duty >= be_lbl),
                "reason": (f"lifetime {lifetime:.0f}d covers the {label} term and duty {duty:.0%} "
                           f">= break-even {be_lbl:.0%}"
                           if duty >= be_lbl else
                           f"duty {duty:.0%} < break-even {be_lbl:.0%} for a {1 - rhr / od_rate:.0%} discount"),
                "break_even_duty": round(be_lbl, 3),
                "discount": round(1.0 - rhr / od_rate, 3),
            })
    else:
        options.append({
            "tier": "reserved", "term": "3yr", "cost": round(hours * res_rate, 2),
            "eligible": False,
            "reason": (f"lifetime {lifetime:.0f}d cannot amortise a 1yr or 3yr commitment"),
            "break_even_duty": round(be, 3), "discount": round(res_discount, 3),
        })
    options.append({
        "tier": "on_demand", "term": "none", "cost": round(hours * od_rate, 2),
        "eligible": True, "reason": "always available, no commitment", "rate": od_rate,
    })

    eligible = [o for o in options if o["eligible"]]
    chosen = min(eligible, key=lambda o: o["cost"])
    naive = min(options, key=lambda o: o["cost"])
    return {
        "tier": chosen["tier"], "term": chosen["term"], "cost": chosen["cost"],
        "reason": chosen["reason"], "options": options, "monthly_gpu_hours": round(hours, 1),
        "duty": round(duty, 3),
        "phantom_savings": round(max(0.0, chosen["cost"] - naive["cost"]), 2),
        "phantom_source": naive["tier"] if naive["cost"] < chosen["cost"] else "",
        "spot_interrupt_breakeven": round(
            spot_interruption_breakeven(hours, spot_rate, od_rate), 4),
        "spot_interrupt_margin": round(
            spot_interruption_breakeven(hours, spot_rate, od_rate) / rate, 2) if rate > 0 else 0.0,
    }


# ===========================================================================
# EXTENSION 3 — cache_is_worth_it(): caching is a *loan*, not free money
#
# A cache write is billed above the normal input price (Anthropic-style 5-min
# TTL: 1.25x). Only the re-reads pay it back. Per 1M tokens of prefix:
#   no cache      : N * price_in
#   with cache    : write_multiplier * price_in + (N - 1) * cache_discount * price_in
# Caching wins when N > (write_multiplier - cache_discount) / (1 - cache_discount).
# With write 1.25x and read 0.10x that break-even is N* = 1.278 reads, i.e. every
# cached prefix must be sent at least TWICE before the discount is real.
# ===========================================================================

CACHE_WRITE_MULTIPLIER = 1.25   # cost of writing a prefix, x price_in
CACHE_READ_DISCOUNT = 0.10      # cost of a cache read, x price_in


def cache_break_even_reads(write_multiplier: float = CACHE_WRITE_MULTIPLIER,
                           read_discount: float = CACHE_READ_DISCOUNT) -> float:
    """Number of sends per cached prefix at which caching breaks even."""
    if read_discount >= 1.0:
        return 0.0
    return max(0.0, (write_multiplier - read_discount) / (1.0 - read_discount))


def cache_is_worth_it(avg_cache_reads: float, write_cost_per_m: float | None = None,
                      read_discount: float = CACHE_READ_DISCOUNT,
                      price_in_per_m: float = 0.0,
                      write_multiplier: float = CACHE_WRITE_MULTIPLIER) -> dict:
    """Decide whether caching a prefix pays for itself, and by how much.

    `avg_cache_reads` is the observed number of sends per cached prefix
    (reads + the original write). `write_cost_per_m` may be supplied directly as
    USD per 1M cached tokens; otherwise it is derived from price_in_per_m x
    write_multiplier (storage/TTL cost). Returns the net saving fraction on the
    input bill, so a caller can audit a savings claim instead of trusting it.
    """
    reads = max(1.0, float(avg_cache_reads))
    write_mult = write_multiplier
    if write_cost_per_m is not None and price_in_per_m > 0:
        write_mult = float(write_cost_per_m) / price_in_per_m
    break_even = cache_break_even_reads(write_mult, read_discount)

    uncached = reads                                        # 1.0 x per send
    cached = write_mult + (reads - 1.0) * read_discount      # write once, read after
    net_fraction = 1.0 - cached / uncached if uncached else 0.0
    naive_fraction = 1.0 - read_discount                     # the "90% off" headline
    return {
        "worth_it": reads > break_even,
        "reads_observed": round(reads, 3),
        "break_even_reads": round(break_even, 3),
        "margin": round(reads / break_even, 2) if break_even > 0 else 0.0,
        "net_saving_frac": round(net_fraction, 4),
        "headline_saving_frac": round(naive_fraction, 4),
        # how much of the advertised discount survives the write cost
        "discount_haircut": round(1.0 - net_fraction / naive_fraction, 4) if naive_fraction > 0 else 0.0,
    }
