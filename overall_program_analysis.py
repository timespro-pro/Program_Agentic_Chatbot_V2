"""
Overall Program Analysis for Version 2.

Builds a deterministic evidence package from program/source rows and optional
advisor-tenure rows. The LLM is used only to explain this evidence, not invent metrics.
"""
from __future__ import annotations

from typing import Any
import math
import pandas as pd


def _num(value):
    try:
        if pd.isna(value):
            return None
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _sum(df, col):
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    return float(s.sum()) if s.notna().any() else None


def _ratio(a, b):
    return a / b if a is not None and b not in (None, 0) else None


def _pct_change(current, previous):
    if current is None or previous in (None, 0):
        return None
    return (current - previous) / abs(previous) * 100


def _program_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    leads = _sum(rows, "leads")
    warm = _sum(rows, "warm_leads")
    apps = _sum(rows, "applications")
    enrols = _sum(rows, "enrollments")
    spend = _sum(rows, "spends")
    # Fee is a program-level value repeated on each source row: never sum it.
    fee = None
    if "total_program_fees_sf" in rows:
        fees = pd.to_numeric(rows["total_program_fees_sf"], errors="coerce").dropna()
        if not fees.empty:
            fee = float(fees.iloc[0])
    plan_budget = _sum(rows, "Planned_Budget")
    plan_leads = _sum(rows, "Planned_Lead")
    plan_apps = _sum(rows, "Planned_Application")
    plan_enrols = _sum(rows, "Planned_Enrollment")
    plan_gsv_crore = _sum(rows, "Planned_GSV")
    # In this dataset Planned_GSV is stored in crore units, while fees are in rupees.
    # Actual GSV is calculated manually as enrolled learners x program fee.
    actual_gsv = enrols * fee if enrols is not None and fee is not None else None
    planned_gsv = plan_gsv_crore * 10_000_000 if plan_gsv_crore is not None else None
    return {
        "leads": leads, "warm_leads": warm, "applications": apps, "enrollments": enrols,
        "spend_observed": spend, "program_fee": fee,
        "L2W": _ratio(warm, leads), "L2A": _ratio(apps, leads),
        "L2E": _ratio(enrols, leads), "W2A": _ratio(apps, warm),
        "A2E": _ratio(enrols, apps),
        "CPL": _ratio(spend, leads), "CAC": _ratio(spend, enrols),
        "planned_budget": plan_budget, "planned_leads": plan_leads,
        "planned_applications": plan_apps, "planned_enrollments": plan_enrols,
        "actual_gsv": actual_gsv,
        "planned_gsv": planned_gsv,
        "planned_gsv_crore_source": plan_gsv_crore,
        "gsv_variance": actual_gsv - planned_gsv if actual_gsv is not None and planned_gsv is not None else None,
        "gsv_plan_attainment_pct": _ratio(actual_gsv, planned_gsv),
        "budget_attainment_pct": _ratio(spend, plan_budget),
        "lead_plan_attainment_pct": _ratio(leads, plan_leads),
        "application_plan_attainment_pct": _ratio(apps, plan_apps),
        "enrollment_plan_attainment_pct": _ratio(enrols, plan_enrols),
    }


def _batch_chain(df: pd.DataFrame, selected: str, max_previous: int = 2):
    codes = set(df["program_code"].dropna().astype(str))
    chain, seen = [], {selected}
    current = selected
    for _ in range(max_previous):
        rows = df[df["program_code"].astype(str).eq(current)]
        if rows.empty or "previous_batch_code" not in rows:
            break
        previous = rows["previous_batch_code"].dropna().astype(str).str.strip()
        previous = next((x for x in previous if x and x.lower() != "nan"), None)
        if not previous or previous in seen:
            break
        seen.add(previous)
        if previous not in codes:
            chain.append({"program_code": previous, "status": "Referenced but not found in data"})
            break
        chain.append({"program_code": previous, "status": "Available"})
        current = previous
    return chain


def _source_table(rows: pd.DataFrame):
    if "source_created" not in rows:
        return []
    work = rows.copy()
    for col in ["leads", "warm_leads", "applications", "enrollments", "spends"]:
        if col in work:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    result = []
    for source, g in work.groupby("source_created", dropna=False):
        leads, warm = _sum(g, "leads"), _sum(g, "warm_leads")
        apps, enrols, spend = _sum(g, "applications"), _sum(g, "enrollments"), _sum(g, "spends")
        result.append({
            "source": str(source) if pd.notna(source) else "Unknown",
            "leads": leads, "warm_leads": warm, "applications": apps, "enrollments": enrols,
            "L2W": _ratio(warm, leads), "L2A": _ratio(apps, leads), "L2E": _ratio(enrols, leads),
            "W2A": _ratio(apps, warm), "A2E": _ratio(enrols, apps),
            "spend_observed": spend, "CPL": _ratio(spend, leads), "CAC": _ratio(spend, enrols),
        })
    return sorted(result, key=lambda x: (x["enrollments"] or 0), reverse=True)


def _advisor_table(advisor_df: pd.DataFrame | None, batch: str):
    if advisor_df is None or advisor_df.empty or "program_code" not in advisor_df:
        return []
    rows = advisor_df[advisor_df["program_code"].astype(str).eq(batch)].copy()
    if rows.empty:
        return []
    out = []
    for tenure, g in rows.groupby("opp_agent_tenure_mth", dropna=False):
        leads, warm = _sum(g, "leads"), _sum(g, "warm")
        apps, enrols = _sum(g, "apps"), _sum(g, "enrols")
        l2w = _ratio(warm, leads)
        l2e = _ratio(enrols, leads)
        # Best-tenure ranking intentionally prioritizes L2W and L2E equally.
        priority_score = (l2w + l2e) / 2 if l2w is not None and l2e is not None else (l2w if l2e is None else l2e)
        out.append({
            "advisor_tenure": str(tenure) if pd.notna(tenure) else "NA / missing",
            "leads": leads, "warm_leads": warm, "applications": apps, "enrollments": enrols,
            "L2W": l2w, "L2A": _ratio(apps, leads), "L2E": l2e,
            "W2A": _ratio(apps, warm), "A2E": _ratio(enrols, apps),
            "tenure_priority_score": priority_score,
        })
    return sorted(out, key=lambda x: (x["tenure_priority_score"] is not None, x["tenure_priority_score"] or -1), reverse=True)


def build_overall_program_analysis(raw_df: pd.DataFrame, program_code: str,
                                   advisor_df: pd.DataFrame | None = None,
                                   selected_old_batches: list[str] | None = None) -> dict[str, Any]:
    if raw_df is None or raw_df.empty or "program_code" not in raw_df:
        return {"error": "Program Batch Data is empty or missing program_code."}
    df = raw_df.copy()
    df["program_code"] = df["program_code"].astype(str).str.strip()
    current_rows = df[df["program_code"].eq(str(program_code))]
    if current_rows.empty:
        return {"error": f"Selected program {program_code} was not found in Program Batch Data."}

    available_codes = set(df["program_code"].dropna().astype(str).str.strip())
    if selected_old_batches is not None:
        # Use only the exact old batches selected in the UI, in selection order,
        # with a hard limit of three historical batches (four total including new).
        selected_codes = []
        for value in selected_old_batches:
            code = str(value).strip()
            if code and code != str(program_code).strip() and code in available_codes and code not in selected_codes:
                selected_codes.append(code)
            if len(selected_codes) >= 3:
                break
        chain = [
            {"program_code": code, "status": "Available"}
            for code in selected_codes
        ]
    else:
        chain = _batch_chain(df, str(program_code), max_previous=3)
    batch_reports = []
    codes = [str(program_code)] + [x["program_code"] for x in chain if x["status"] == "Available"]
    for code in codes:
        rows = df[df["program_code"].eq(code)]
        metrics = _program_metrics(rows)
        # Aggregate source rows, but report plan fields only when present.
        batch_reports.append({
            "program_code": code,
            "is_selected_batch": code == str(program_code),
            "metrics": metrics,
            "source_performance": _source_table(rows) if code == str(program_code) else [],
            "advisor_tenure_mix": _advisor_table(advisor_df, code),
        })

    current = batch_reports[0]
    previous = batch_reports[1:] 
    comparisons = []
    for old in previous:
        comp = {"program_code": old["program_code"], "status": "Available", "metrics": {}}
        for key in ["leads", "warm_leads", "applications", "enrollments", "L2W", "L2A", "L2E", "W2A", "A2E", "CPL", "CAC", "program_fee"]:
            cur_v, old_v = current["metrics"].get(key), old["metrics"].get(key)
            comp["metrics"][key] = {
                "current": cur_v, "previous": old_v,
                "absolute_change": cur_v - old_v if cur_v is not None and old_v is not None else None,
                "percent_change": _pct_change(cur_v, old_v),
            }
        comparisons.append(comp)

    return {
        "program_code": str(program_code),
        "status": current_rows["program_status_f"].dropna().astype(str).iloc[0] if "program_status_f" in current_rows and current_rows["program_status_f"].notna().any() else None,
        "category": current_rows["category"].dropna().astype(str).iloc[0] if "category" in current_rows and current_rows["category"].notna().any() else None,
        "program_fee_note": "total_program_fees_sf is a program-level value repeated across source rows; treated as one value, not summed.",
        "gsv_note": "Actual GSV is manually calculated as enrollments × total_program_fees_sf. Planned_GSV source values are in crore units and are converted to rupees for like-for-like comparison.",
        "batch_chain": chain,
        "batches": batch_reports,
        "comparisons_to_previous_batches": comparisons,
        "selected_old_batches": [x["program_code"] for x in chain if x["status"] == "Available"],
        "selection_mode": "user_selected" if selected_old_batches is not None else "automatic_lineage",
        "decision_rule": "Evaluate plan attainment and compare the current batch with every selected historical batch provided in the evidence. If historical batches are unavailable, state that limitation.",
        "metric_definitions": {
            "L2W": "warm leads / leads", "L2A": "applications / leads", "L2E": "enrollments / leads",
            "W2A": "applications / warm leads", "A2E": "enrollments / applications",
            "CPL": "observed spend / leads", "CAC": "observed spend / enrollments",
        },
    }


def format_overall_program_fallback(result: dict[str, Any]) -> str:
    """Readable deterministic fallback if the LLM is unavailable."""
    if result.get("error"):
        return result["error"]
    current = result["batches"][0]
    m = current["metrics"]
    def fmt(x, pct=False):
        if x is None: return "N/A"
        return f"{x*100:.1f}%" if pct else f"{x:,.2f}"
    lines = [
        f"# Overall Program Analysis — {result['program_code']}",
        "## Executive diagnosis",
        "Overall assessment should be based on plan attainment and the two available historical comparisons. The concise reason summary and supporting detail follow below.",
        "## Planned versus actual",
        "| Metric | Plan | Actual |",
        "|---|---:|---:|",
    ]
    for label, plan_key, actual_key in [
        ("Leads", "planned_leads", "leads"),
        ("Applications", "planned_applications", "applications"),
        ("Enrollments", "planned_enrollments", "enrollments"),
        ("Budget / observed spend", "planned_budget", "spend_observed"),
        ("GSV (₹)", "planned_gsv", "actual_gsv"),
    ]:
        lines.append(f"| {label} | {fmt(m.get(plan_key))} | {fmt(m.get(actual_key))} |")
    lines += ["## Funnel and efficiency metrics", "| Metric | Actual |", "|---|---:|"]
    for label, key in [("Warm leads", "warm_leads"), ("L2W", "L2W"), ("L2A", "L2A"), ("L2E", "L2E"), ("W2A", "W2A"), ("A2E", "A2E"), ("CPL", "CPL"), ("CAC", "CAC"), ("Program fee", "program_fee")]:
        lines.append(f"| {label} | {fmt(m.get(key), key in {"L2W","L2A","L2E","W2A","A2E"})} |")
    prior_batches = result["batches"][1:3]
    prev1 = prior_batches[0]["program_code"] if prior_batches else "Previous batch"
    prev2 = prior_batches[1]["program_code"] if len(prior_batches) > 1 else "Previous-to-previous"
    lines += ["## 3. Current vs previous batch", "Legend: green ↑ = positive movement · red ↓ = negative movement. Conversion-rate changes are percentage points; other changes are percentages.",
              "### 3.1 Volume and commercial outcomes",
              f"| Metric | Current batch | Change vs {prev1} | {prev1} | Change vs {prev2} | {prev2} |",
              "|---|---:|---:|---:|---:|---:|"]
    volume_keys = [("Leads", "leads", True), ("Warm leads", "warm_leads", True), ("Applications", "applications", True),
                   ("Enrollments", "enrollments", True), ("Program fee (₹)", "program_fee", None), ("Actual GSV (₹)", "actual_gsv", True)]
    def trend_marker(key, change_value):
        # Direction follows the sign shown in the comparison cell: positive = green up,
        # negative = red down. Program fee is a contextual price point, so it stays neutral.
        if change_value is None or key == "program_fee" or change_value == 0:
            return ""
        return "[[UP]] " if change_value > 0 else "[[DOWN]] "
    for label, key, favorable_higher in volume_keys:
        cells = [label, fmt(m.get(key))]
        for old in prior_batches:
            old_m = old["metrics"]
            cur_v, old_v = m.get(key), old_m.get(key)
            if cur_v is None or old_v in (None, 0):
                change, delta = "N/A", None
            else:
                delta = (cur_v-old_v)/abs(old_v)*100
                change = f"{delta:+.1f}%"
            cells.extend([trend_marker(key, delta) + change, fmt(old_m.get(key))])
        while len(cells) < 6: cells.extend(["N/A", "N/A"])
        lines.append("| " + " | ".join(cells[:6]) + " |")
    lines += ["### 3.2 Conversion and acquisition efficiency",
              f"| Metric | Current batch | Change vs {prev1} | {prev1} | Change vs {prev2} | {prev2} |",
              "|---|---:|---:|---:|---:|---:|"]
    efficiency_keys = [("L2W", "L2W", True), ("L2A", "L2A", True), ("L2E", "L2E", True), ("W2A", "W2A", True),
                       ("A2E", "A2E", True), ("CPL", "CPL", False), ("CAC", "CAC", False)]
    for label, key, is_rate in efficiency_keys:
        cells = [label, fmt(m.get(key), is_rate)]
        for old in prior_batches:
            old_m = old["metrics"]
            cur_v, old_v = m.get(key), old_m.get(key)
            if cur_v is None or old_v in (None, 0):
                change, delta = "N/A", None
            elif is_rate:
                delta = (cur_v-old_v)*100
                change = f"{delta:+.1f} pp"
            else:
                delta = (cur_v-old_v)/abs(old_v)*100
                change = f"{delta:+.1f}%"
            cells.extend([trend_marker(key, delta) + change, fmt(old_m.get(key), is_rate)])
        while len(cells) < 6: cells.extend(["N/A", "N/A"])
        lines.append("| " + " | ".join(cells[:6]) + " |")
    lines += ["## Source-wise root cause (3–4 line summary)", "See source scorecard below; prioritize sources with adequate lead volume but lower L2A, then inspect L2W and W2A to locate the funnel drop.", 
              "| Source | Leads | Warm | Apps | Enrols | L2W | L2A | W2A | A2E | CPL | CAC |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in current.get("source_performance", []):
        lines.append("| " + " | ".join([s["source"]] + [fmt(s.get(k), k in {"L2W","L2A","L2E","W2A","A2E"}) for k in ["leads","warm_leads","applications","enrollments","L2W","L2A","W2A","A2E","CPL","CAC"]]) + " |")
    if current.get("advisor_tenure_mix"):
        lines += ["## Advisor tenure mix (ranked by L2W and L2E)", "| Tenure | Leads | Warm | Apps | Enrols | L2W | L2E | Priority score |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for s in current["advisor_tenure_mix"]:
            lines.append("| " + " | ".join([s["advisor_tenure"]] + [fmt(s.get(k), k in {"L2W","L2E"}) for k in ["leads","warm_leads","applications","enrollments","L2W","L2E","tenure_priority_score"]]) + " |")
        lines.append("")
        lines.append("Experience interpretation: Lower tenure-month values represent more experienced advisors, while higher values represent less experienced advisors. Use the L2W/L2E ranking and lead volumes above to identify which experience groups perform better in this dataset; this is an association, not proof that tenure causes the difference.")
    return "\n".join(lines)


def build_section_performance_statuses(result: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Create transparent Good/Bad/Mixed verdicts from computed metrics only."""
    if not result or result.get("error") or not result.get("batches"):
        return {}

    batches = result["batches"]
    current = batches[0].get("metrics", {})
    previous = [b.get("metrics", {}) for b in batches[1:]]
    statuses: dict[str, dict[str, str]] = {}

    def usable(values):
        return [float(v) for v in values if v is not None and math.isfinite(float(v))]

    def verdict(good_count: int, bad_count: int, total: int) -> str:
        if total <= 0:
            return "MIXED"
        if bad_count >= 2 and bad_count > good_count:
            return "BAD"
        if good_count >= 2 and good_count > bad_count:
            return "GOOD"
        return "MIXED"

    # Plan vs actual: judge attainment with higher-is-better volume/GSV and
    # lower-is-better budget/cost where the relevant plan exists.
    plan_checks = []
    for key in ["lead_plan_attainment_pct", "application_plan_attainment_pct",
                "enrollment_plan_attainment_pct", "gsv_plan_attainment_pct"]:
        value = current.get(key)
        if value is not None:
            plan_checks.append(value >= 0.95)
    # Budget overrun is a caution, not automatically bad if outcomes exceed plan.
    budget = current.get("budget_attainment_pct")
    if budget is not None:
        outcomes = usable([current.get("application_plan_attainment_pct"),
                           current.get("enrollment_plan_attainment_pct"),
                           current.get("gsv_plan_attainment_pct")])
        plan_checks.append(not (budget > 1.10 and (not outcomes or sum(outcomes) / len(outcomes) < 0.95)))
    good = sum(plan_checks)
    bad = len(plan_checks) - good
    statuses["Plan vs Actual"] = {
        "status": verdict(good, bad, len(plan_checks)),
        "reason": "Based on available plan-attainment ratios; missing plan values are excluded."
    }

    # Historical volume and commercial outcomes.
    volume_keys = ["leads", "warm_leads", "applications", "enrollments", "actual_gsv"]
    changes = []
    for prev in previous:
        for key in volume_keys:
            a, b = current.get(key), prev.get(key)
            if a is not None and b not in (None, 0):
                changes.append((a - b) / abs(b))
    good = sum(delta >= 0.03 for delta in changes)
    bad = sum(delta <= -0.03 for delta in changes)
    statuses["Volume and commercial outcomes"] = {
        "status": verdict(good, bad, good + bad),
        "reason": "Based on changes in volume and GSV against the selected historical batches."
    }

    # Conversion/acquisition: rates should rise; CPL/CAC should fall.
    eff_checks = []
    for prev in previous:
        for key in ["L2W", "L2A", "L2E", "W2A", "A2E"]:
            a, b = current.get(key), prev.get(key)
            if a is not None and b is not None:
                eff_checks.append((a - b) >= -0.001)
        for key in ["CPL", "CAC"]:
            a, b = current.get(key), prev.get(key)
            if a is not None and b is not None:
                eff_checks.append(a <= b * 1.03)
    good = sum(eff_checks)
    bad = len(eff_checks) - good
    statuses["Conversion and acquisition efficiency"] = {
        "status": verdict(good, bad, len(eff_checks)),
        "reason": "Conversion rates are assessed as higher-is-better; CPL and CAC as lower-is-better."
    }

    # Source-wise analysis: compare meaningful source conversion spread and
    # observed spend coverage. Mixed is intentional where source evidence is sparse.
    sources = current.get("source_performance", [])
    source_rates = usable([s.get("L2A") for s in sources])
    spend_count = sum(s.get("spend") is not None or s.get("CPL") is not None or s.get("CAC") is not None for s in sources)
    if len(source_rates) >= 2 and max(source_rates) > 0:
        spread = max(source_rates) - min(source_rates)
        source_status = "MIXED" if spread > 0.01 or spend_count < len(sources) else "GOOD"
        source_reason = "Source conversion performance varies; spend coverage is considered when judging efficiency."
    else:
        source_status = "MIXED"
        source_reason = "Insufficient source-level variation or data for a confident overall verdict."
    statuses["Source-wise root-cause analysis"] = {"status": source_status, "reason": source_reason}

    # Advisor tenure: only assess when enough groups have data; do not equate
    # tenure itself with performance.
    advisors = current.get("advisor_tenure_mix", [])
    advisor_l2e = usable([x.get("L2E") for x in advisors])
    if len(advisor_l2e) >= 2:
        spread = max(advisor_l2e) - min(advisor_l2e)
        advisor_status = "MIXED" if spread > 0.01 else "GOOD"
        advisor_reason = "Based on observed conversion differences across available advisor-tenure groups, not tenure alone."
    else:
        advisor_status = "MIXED"
        advisor_reason = "Insufficient advisor-tenure group data for a confident verdict."
    statuses["Advisor-tenure mix"] = {"status": advisor_status, "reason": advisor_reason}
    return statuses

