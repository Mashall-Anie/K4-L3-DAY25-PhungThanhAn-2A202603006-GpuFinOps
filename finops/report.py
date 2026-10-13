"""Report assembly — the lab's deliverable: baseline vs optimized + savings chart."""
from __future__ import annotations


def _money(x: float) -> str:
    return f"${x:,.0f}"


def build_report(baseline_usd: float, optimized_usd: float, levers: dict,
                 sustainability: dict | None = None, period: str = "monthly",
                 unit_economics: dict | None = None, levers_detail: dict | None = None,
                 sections: list | None = None,
                 title: str = "NimbusAI — GPU Cost Optimization Report") -> str:
    """Return a markdown cost-optimization report.

    The baseline signature (baseline, optimized, levers) keeps producing the
    headline numbers; the optional arguments add the unit economics, the
    per-lever narrative, and any extra analyses the caller wants included.
    """
    savings = baseline_usd - optimized_usd
    pct = (savings / baseline_usd * 100.0) if baseline_usd > 0 else 0.0
    lines = [
        f"# {title}",
        "",
        f"**Period:** {period}  ",
        f"**Baseline spend:** {_money(baseline_usd)}  ",
        f"**Optimized spend:** {_money(optimized_usd)}  ",
        f"**Projected savings:** {_money(savings)}  (**{pct:.0f}%**)",
        "",
    ]

    if unit_economics:
        lines += ["## Unit economics — the number that matters", "",
                  "| Metric | Baseline | Optimized | Change |", "|---|---|---|---|"]
        for name, ue in unit_economics.items():
            before, after = ue.get("baseline"), ue.get("optimized")
            delta = ue.get("delta", "")
            fmt = ue.get("fmt", "{:,.3f}")
            lines.append(f"| {name} | {fmt.format(before)} | {fmt.format(after)} | {delta} |")
        lines.append("")

    lines += ["## Savings by lever", "", "| Lever | Savings (USD) | Share of savings |", "|---|---|---|"]
    for name, amount in levers.items():
        share = (amount / savings * 100.0) if savings > 0 else 0.0
        lines.append(f"| {name} | {_money(amount)} | {share:.0f}% |")
    lines.append(f"| **Total** | **{_money(savings)}** | **100%** |")
    lines.append("")

    if levers_detail:
        lines += ["### What each lever does", ""]
        for name, detail in levers_detail.items():
            if isinstance(detail, dict):
                lines.append(f"**{name}** — {detail.get('what', '')}  ")
                lines.append(f"Evidence: {detail.get('evidence', '')}  ")
                lines.append(f"Why it works: {detail.get('why', '')}")
            else:
                lines.append(f"**{name}** — {detail}")
            lines.append("")

    if sustainability:
        lines += [
            "## Sustainability (energy & carbon)", "",
            f"- Energy per query: {sustainability.get('wh_per_query', 0):.2f} Wh "
            f"({sustainability.get('queries_per_kwh', 0):,.0f} queries/kWh)",
            f"- Carbon per query: {sustainability.get('carbon_g', 0):.3f} gCO2e "
            f"({sustainability.get('carbon_kg_per_day', 0):,.1f} kgCO2e/day at "
            f"{sustainability.get('carbon_region', 'us-east-1')})",
            f"- Cleanest region: {sustainability.get('cleanest_region', 'n/a')} "
            f"({sustainability.get('cleanest_region_gco2_kwh', 0):.0f} gCO2e/kWh)",
            f"- Best cost+carbon blend: {sustainability.get('best_region', 'n/a')} "
            f"({sustainability.get('best_region_gco2_kwh', 0):.0f} gCO2e/kWh, "
            f"${sustainability.get('best_region_kwh_price', 0):.3f}/kWh)",
            f"- Carbon-aware move of interruptible jobs: "
            f"{sustainability.get('carbon_saved_kg', 0):,.0f} kgCO2e saved "
            f"({sustainability.get('carbon_saved_pct', 0):.0f}%) and "
            f"${sustainability.get('energy_cost_saved_usd', 0):,.0f}/month of electricity, "
            f"by relocating {sustainability.get('jobs_moved', 0)} jobs to "
            f"{sustainability.get('to_region', 'n/a')}",
        ]
        extra = sustainability.get("notes", [])
        lines += [f"- {n}" for n in extra]
        lines.append("")

    for section in sections or []:
        block = section.rstrip()
        if block:
            lines += [block, ""]

    lines.append("_Figures are June-2026 as-of snapshots; re-baseline before acting._")
    return "\n".join(lines)


def savings_waterfall(levers: dict, path: str, baseline_usd: float | None = None,
                      optimized_usd: float | None = None,
                      unit_economics: dict | None = None) -> str:
    """Write the savings chart PNG. Returns the path. No-op if matplotlib absent.

    Two panels: a waterfall from baseline spend down through each lever (the
    question "where did the money go?"), and the $/1M-token before/after bars
    (the question "are we serving cheaper?").
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""

    names = list(levers.keys())
    vals = [levers[n] for n in names]
    total = sum(vals)
    if baseline_usd is None:
        baseline_usd = total
    if optimized_usd is None:
        optimized_usd = baseline_usd - total

    has_units = bool(unit_economics)
    if has_units:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [2.1, 1]})
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax2 = None

    # --- waterfall: baseline -> each lever -> optimized ---
    labels = ["Baseline"] + [n.split(" (")[0] for n in names] + ["Optimized"]
    running = baseline_usd
    bottoms, heights, colors = [], [], []
    bottoms.append(0.0)
    heights.append(baseline_usd)
    colors.append("#444b55")
    for v in vals:
        bottoms.append(running - v)
        heights.append(v)
        colors.append("#c0392b")
        running -= v
    bottoms.append(0.0)
    heights.append(running)
    colors.append("#2e7d32")

    x = range(len(labels))
    ax.bar(x, heights, bottom=bottoms, color=colors, width=0.62)
    for i, (b, h) in enumerate(zip(bottoms, heights)):
        ax.text(i, b + h + baseline_usd * 0.015, f"${h:,.0f}", ha="center", fontsize=8.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("USD / month")
    ax.set_title("GPU spend waterfall — baseline to optimized", fontsize=11)
    ax.set_ylim(0, baseline_usd * 1.16)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    # --- unit economics panel ---
    if ax2 is not None:
        names_u = list(unit_economics.keys())
        before = [unit_economics[n].get("baseline", 0) for n in names_u]
        after = [unit_economics[n].get("optimized", 0) for n in names_u]
        pos = range(len(names_u))
        w = 0.36
        ax2.bar([p - w / 2 for p in pos], before, width=w, label="baseline", color="#444b55")
        ax2.bar([p + w / 2 for p in pos], after, width=w, label="optimized", color="#2e7d32")
        for p, (b, a) in enumerate(zip(before, after)):
            ax2.text(p - w / 2, b, f"{b:,.2f}", ha="center", va="bottom", fontsize=8)
            ax2.text(p + w / 2, a, f"{a:,.2f}", ha="center", va="bottom", fontsize=8)
        ax2.set_xticks(list(pos))
        ax2.set_xticklabels([n.replace(" — ", "\n") for n in names_u], fontsize=8.5)
        ax2.set_title("Cost per 1M tokens served", fontsize=11)
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax2.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)
    return path
