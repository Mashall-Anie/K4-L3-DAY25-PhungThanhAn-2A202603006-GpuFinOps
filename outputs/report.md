# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $14,705  
**Projected savings:** $12,428  (**46%**)

## Unit economics — the number that matters

| Metric | Baseline | Optimized | Change |
|---|---|---|---|
| Inference API — $/1M-token | $6.488 | $1.126 | 83% lower |
| All-in incl. GPU rental — $/1M-token | $120.06 | $65.07 | 46% lower |

## Savings by lever

| Lever | Savings (USD) | Share of savings |
|---|---|---|
| Inference (cascade/cache/batch) | $1,212 | 10% |
| Purchasing (spot/reserved) | $10,105 | 81% |
| Right-size over-provisioned GPUs | $511 | 4% |
| Kill idle GPUs | $600 | 5% |
| **Total** | **$12,428** | **100%** |

### What each lever does

**Inference (cascade/cache/batch)** — route simple traffic to the small model, bill cached input at 0.1x, and batch everything that can wait  
Evidence: $6.488 -> $1.126 per 1M tokens (82.6%) on 7,533,027 tokens/day  
Why it works: the levers multiply, they do not add: batch (x0.50) x full cache hit (x0.10) = 0.05 of the naive bill (stack = 0.05)

**Purchasing (spot/reserved)** — spot + checkpoint for interruptible jobs, reserved for services that really run at duty cycle above break-even  
Evidence: $25,667 -> $15,562 per month (39.4% off on-demand)  
Why it works: the workload mix is ['reserved', 'spot']: reserved discounts run 20-44% by card, and short batch jobs ride spot instead of locking a commitment longer than their own life

**Right-size over-provisioned GPUs** — move GPUs that never approach either ceiling onto cheaper silicon  
Evidence: gpu-h100-4 H100->A100 (peak needs 1.59 TB/s, 301 TF, 67 GB) saves $511/mo  
Why it works: you rent the card, not the FLOPs: if neither MFU nor MBU is near target, a smaller card does the same job for less rent

**Kill idle GPUs** — shut down or reschedule the GPU-hours where the card is <10% utilised  
Evidence: 8 idle h/day on gpu-h100-5 ($20.00/day, $600/month)  
Why it works: an idle reserved GPU is the most expensive kind: you pay the full hour and get no tokens for it

## Sustainability (energy & carbon)

- Energy per query: 0.24 Wh (4,167 queries/kWh)
- Carbon per query: 0.091 gCO2e (42.9 kgCO2e/day at us-east-1)
- Cleanest region: europe-north1 (30 gCO2e/kWh)
- Best cost+carbon blend: us-east-wa (90 gCO2e/kWh, $0.055/kWh)
- Carbon-aware move of interruptible jobs: 519 kgCO2e saved (76%) and $116/month of electricity, by relocating 5 jobs to us-east-wa
- Reasoning traffic is 8.4% of requests but 94.0% of inference energy (148 Wh/query vs 0.86 Wh/query).
- Cutting idle + right-sizing removes ~384 kWh/month from the fleet.
- Electricity price and grid carbon move together here: the cleanest blend is also the cheapest, so the carbon decision is free money (see Extension 5).

## Why GPU-Util is a lie (and what it costs)

`nvidia-smi` reports *time-active clocks*, not work done. On gpu-h100-4 and gpu-a10g-1 the card reported 98% util while MFU was only 0.19 and 97% util while MFU was only 0.27. At $2.50/GPU-hour the fleet pays for a full H100 hour and receives about 19% of the FLOPs, 21% of the bandwidth. Financial impact: $511/month of the bill buys silicon the workload cannot use (right-sizing lever), on top of $600/month burned on idle hours. The fix is not 'make the GPU busier', it is to measure MFU/MBU and buy the card the job actually needs.

## Your Turn extensions (measured)

### Extension 1 — risk-adjusted tier policy (`pricing.tier_decision`)

- Prices spot/reserved/on-demand on each job's own hours, with per-GPU-type interruption rates (H100 3%, H200 3%, B200 4%, ...).

- Result: $15,627 -> $15,562/month ($+65 vs the simple policy) and it **refuses $2,083/month of phantom savings** from commitments no job could amortise.

- Real cost of interruption risk: 352 GPU-hours of rework ($451/month) — cheap insurance against a 1-3 year lock-in.

- Flexibility price: 1yr instead of 3yr on the persistent services costs $2,232/month more.

### Extension 2 — MBU-driven right-sizing (`metrics.rightsize_gpu`)

- Rule: flag when peak MFU < 0.30 **and** peak MBU < 0.60; then pick the cheapest card that still covers observed peak FLOPs, peak bandwidth (at <=60% MBU) and peak HBM.

- Result: 1 GPU(s) re-shaped for $511/month (-216 kWh/month) with no job moved off its workload.

### Extension 3 — `cache_is_worth_it()` (cache is a loan, not free money)

- Observed reuse: 1,703,990 cached tokens read over 195,335 written -> **8.72 sends per prefix** vs break-even 1.28.

- The advertised 90% off is really **76.8% off** once the 1.25x write price is paid (haircut 14.6%); audited cache value $27/month.

- 57 sessions (19.3% of traffic) are single-send: cache must be **off** there or it loses money.

### Extension 4 — reasoning budget

- Reasoning is 8.4% of requests but 35.6% of output tokens and 16.5% of the optimized bill.

- Energy: **94.0% of inference energy** (148 Wh/query vs 0.86 Wh/query).

- Rule 'reasoning only on the large tier, everything else capped at 641 output tokens': 162 requests capped, $3/month and 379.9 kWh/month avoided (correctness stays with the big model).

### Extension 5 — carbon-aware scheduling

- Move 5 interruptible jobs us-east-1 -> us-east-wa: 1,789 kWh gives 680 -> 161 kgCO2e (**-76%**) and $215 -> $98 of electricity (**-$116/month**).

- `europe-north1` is the cleanest grid (30 gCO2e/kWh); `us-east-wa` wins the cost+carbon blend. Same workload, different postcode, different bill.

## Cost allocation: showback before chargeback

- Team showback ($2.59/day for the biggest spender, assistant) is only trustworthy at **92% tag coverage** -> chargeback gate: **OPEN**.

- $0.70/day of spend still has no `project` tag (197 of 2,400 requests): that slice cannot be charged back to a project until tagging is fixed.

- `outputs/focus_export.csv` carries all 2,400 request-level rows (BilledCost + team/project + SkuId) so the same feed works for showback, chargeback and multi-vendor roll-ups.

## Recommended actions, in order

1. **Close the purchasing gap first ($10,105/month).** Largest, safest lever: reserve only services measured at duty >= break-even, spot everything checkpointable, and never commit to a term longer than the workload's life.

2. **Re-route inference traffic ($1,212/month).** Small-model default, cache only prefixes reused >= 2 times, batch what can wait. Then attack reasoning energy (94% of inference Wh).

3. **Fix the fleet shape ($1,111/month + carbon).** Right-size the GPUs that never approach MFU/MBU targets, kill the idle hours (8 h/day on gpu-h100-5), and schedule interruptible jobs in us-east-wa for -76% CO2e at no extra cost.

## Caveats

- Prices are a June-2026 snapshot; re-baseline before committing money.

- Purchasing economics are modelled as a 30-day run rate per job (the mission's monthly convention), not per individual run.

- The reasoning energy model (80x multiplier on token counts) is directional: at this traffic volume it implies more energy than the 11-GPU telemetry window shows, so treat Wh as relative, not absolute.

_Figures are June-2026 as-of snapshots; re-baseline before acting._