"""Deterministic multi-batch analysis for one program's batch lineage.

Uses the actual rows for each historical batch in Batch_wise_program_data.csv.
It does not sum previous_batch_* benchmark fields across source rows.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

PROGRAM_COL = "program_code"
PREVIOUS_COL = "previous_batch_code"
SOURCE_COL = "source_created"
ACTUAL_METRICS = ["leads", "warm_leads", "applications", "enrollments", "gsv_cr", "spends"]
PLAN_MAP = {
    "spends": "Planned_Budget",
    "leads": "Planned_Lead",
    "applications": "Planned_Application",
    "enrollments": "Planned_Enrollment",
    "gsv_cr": "Planned_GSV",
}
PAID_MAP = {
    "paid_leads": "planned_paid_leads",
    "paid_enrollments": "planned_paid_enrollments",
    "paid_gsv_cr": "planned_paid_gsv",
    "paid_spends": "planned_paid_spends",
}


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[₹,$,%\s]", "", regex=True).str.replace(",", "", regex=False).replace({"nan": np.nan, "None": np.nan, "": np.nan}), errors="coerce")


def _sum_valid(frame: pd.DataFrame, column: str):
    if column not in frame.columns:
        return None
    vals = _numeric(frame[column])
    if vals.notna().sum() == 0:
        return None
    return float(vals.sum(min_count=1))


def _ratio(n, d):
    if n is None or d is None or pd.isna(n) or pd.isna(d) or d == 0:
        return None
    return float(n / d)


def _first_valid(frame: pd.DataFrame, column: str):
    if column not in frame.columns:
        return None
    vals = frame[column].dropna()
    if vals.empty:
        return None
    return vals.iloc[0]


def _metric_snapshot(frame: pd.DataFrame) -> dict[str, Any]:
    out = {m: _sum_valid(frame, m) for m in ACTUAL_METRICS}
    for m, plan in PLAN_MAP.items():
        out[f"plan_{m}"] = _sum_valid(frame, plan)
    # Paid fields are program-level and repeated on each source row. Deduplicate
    # only when all non-null values agree; never sum them over source rows.
    for col in ["paid_leads", "paid_enrollments", "paid_gsv_cr", "paid_spends", *PAID_MAP.values()]:
        if col not in frame.columns:
            out[col] = None
            continue
        vals = _numeric(frame[col]).dropna().unique()
        out[col] = float(vals[0]) if len(vals) == 1 else None
        if len(vals) > 1:
            out.setdefault("warnings", []).append(f"Conflicting repeated program-level values in {col}; treated as unavailable.")
    out["cpl"] = _ratio(out.get("spends"), out.get("leads"))
    out["cost_per_application"] = _ratio(out.get("spends"), out.get("applications"))
    out["cac"] = _ratio(out.get("spends"), out.get("enrollments"))
    out["lead_to_warm_pct"] = 100 * _ratio(out.get("warm_leads"), out.get("leads")) if _ratio(out.get("warm_leads"), out.get("leads")) is not None else None
    out["lead_to_application_pct"] = 100 * _ratio(out.get("applications"), out.get("leads")) if _ratio(out.get("applications"), out.get("leads")) is not None else None
    out["warm_to_application_pct"] = 100 * _ratio(out.get("applications"), out.get("warm_leads")) if _ratio(out.get("applications"), out.get("warm_leads")) is not None else None
    out["application_to_enrollment_pct"] = 100 * _ratio(out.get("enrollments"), out.get("applications")) if _ratio(out.get("enrollments"), out.get("applications")) is not None else None
    out["lead_to_enrollment_pct"] = 100 * _ratio(out.get("enrollments"), out.get("leads")) if _ratio(out.get("enrollments"), out.get("leads")) is not None else None
    out["gsv_per_enrollment_cr"] = _ratio(out.get("gsv_cr"), out.get("enrollments"))
    out["gsv_to_spend"] = _ratio(out.get("gsv_cr") * 10_000_000 if out.get("gsv_cr") is not None else None, out.get("spends"))
    out["paid_cpl"] = _ratio(out.get("paid_spends"), out.get("paid_leads"))
    out["paid_cac"] = _ratio(out.get("paid_spends"), out.get("paid_enrollments"))
    out["paid_gsv_to_spend"] = _ratio(out.get("paid_gsv_cr") * 10_000_000 if out.get("paid_gsv_cr") is not None else None, out.get("paid_spends"))
    # Program fees are repeated across source rows; treat them as a program-level value.
    fee_values = _numeric(frame["total_program_fees"]).dropna().unique() if "total_program_fees" in frame.columns else []
    out["program_fees"] = float(fee_values[0]) if len(fee_values) == 1 else None
    if len(fee_values) > 1:
        out.setdefault("warnings", []).append("Conflicting total_program_fees values within this batch; price-point analysis is unavailable for this batch.")
    # GSV identity check: use fee × enrollments only when observed GSV agrees with it.
    expected_gsv_cr = (out["program_fees"] * out.get("enrollments") / 10_000_000) if out.get("program_fees") is not None and out.get("enrollments") is not None else None
    observed_gsv = out.get("gsv_cr")
    out["fee_implied_gsv_cr"] = expected_gsv_cr
    out["gsv_fee_reconciliation_delta_cr"] = (observed_gsv - expected_gsv_cr) if observed_gsv is not None and expected_gsv_cr is not None else None
    out["row_count"] = int(len(frame))
    out["source_count"] = int(frame[SOURCE_COL].nunique(dropna=True)) if SOURCE_COL in frame.columns else None
    out["spend_coverage_rows"] = int(_numeric(frame["spends"]).notna().sum()) if "spends" in frame.columns else 0
    out["spend_missing_rows"] = int(_numeric(frame["spends"]).isna().sum()) if "spends" in frame.columns else len(frame)
    return out


def get_batch_lineage(df: pd.DataFrame, program_code: str, max_batches: int = 4) -> list[str]:
    """Return selected batch and up to max_batches-1 available ancestors, newest first."""
    if PROGRAM_COL not in df.columns:
        raise ValueError(f"Required column missing: {PROGRAM_COL}")
    programs = set(df[PROGRAM_COL].dropna().astype(str).str.strip())
    if program_code not in programs:
        raise ValueError(f"Program {program_code!r} was not found in the dataset.")
    parent = {}
    for p, g in df.groupby(PROGRAM_COL, dropna=True):
        vals = g[PREVIOUS_COL].dropna().astype(str).str.strip().unique() if PREVIOUS_COL in g.columns else []
        parent[str(p).strip()] = vals[0] if len(vals) == 1 else None
    lineage, seen, current = [], set(), program_code
    while current and current in programs and current not in seen:
        lineage.append(current); seen.add(current)
        current = parent.get(current)
    return lineage[:max(1, int(max_batches))]


def analyze_batch_history(
    df: pd.DataFrame,
    program_code: str,
    max_batches: int = 4,
    selected_old_batches: list[str] | None = None,
) -> dict[str, Any]:
    """Analyze the new batch against exactly the old batches selected by the user.

    When selected_old_batches is supplied, all selected, existing program codes are
    included (even if a previous_batch_code pointer skips one). Otherwise, retain
    the default lineage behavior for backward compatibility.
    """
    clean = df.copy()
    clean[PROGRAM_COL] = clean[PROGRAM_COL].astype(str).str.strip()
    available_codes = set(clean[PROGRAM_COL].dropna().astype(str).str.strip())
    full_lineage = get_batch_lineage(clean, program_code, max_batches=max(1, len(available_codes)))
    if selected_old_batches is not None:
        selected = []
        for code in selected_old_batches:
            code = str(code).strip()
            if code and code != program_code and code in available_codes and code not in selected:
                selected.append(code)
        lineage = [program_code] + selected
    else:
        lineage = full_lineage[:max(1, int(max_batches))]
    batches = []
    for code in lineage:
        frame = clean[clean[PROGRAM_COL] == code].copy()
        snap = _metric_snapshot(frame)
        raw_previous = _first_valid(frame, PREVIOUS_COL)
        available_codes = set(clean[PROGRAM_COL].dropna().astype(str).str.strip())
        previous_available = str(raw_previous).strip() in available_codes if raw_previous is not None and not pd.isna(raw_previous) else False
        batches.append({
            "program_code": code,
            # Preserve the raw reference for audit, but explicitly distinguish a missing batch.
            "previous_batch_code": str(raw_previous).strip() if raw_previous is not None and not pd.isna(raw_previous) else None,
            "previous_batch_available": previous_available,
            "previous_batch_display": (str(raw_previous).strip() if previous_available else (f"{str(raw_previous).strip()} (not in dataset)" if raw_previous is not None and not pd.isna(raw_previous) else "N/A")),
            "sales_start_date": _first_valid(frame, "sales_start_date"),
            "category": _first_valid(frame, "category"),
            "financial_year": _first_valid(frame, "financial_year"),
            "status": _first_valid(frame, "program_status_f"),
            "metrics": snap,
        })
    comparisons = []
    if batches:
        current = batches[0]
        current_m = current["metrics"]
        for hist in batches[1:]:
            old_m = hist["metrics"]
            row = {"current_batch": current["program_code"], "historical_batch": hist["program_code"]}
            for metric in ACTUAL_METRICS:
                a, b = current_m.get(metric), old_m.get(metric)
                row[f"{metric}_current"] = a
                row[f"{metric}_historical"] = b
                row[f"{metric}_delta"] = a - b if a is not None and b is not None else None
                row[f"{metric}_change_pct"] = ((a - b) / abs(b) * 100) if a is not None and b not in (None, 0) else None
            for metric in ["cpl", "cost_per_application", "cac", "lead_to_warm_pct", "lead_to_application_pct", "warm_to_application_pct", "application_to_enrollment_pct", "lead_to_enrollment_pct", "gsv_to_spend", "paid_cpl", "paid_cac", "paid_gsv_to_spend"]:
                a, b = current_m.get(metric), old_m.get(metric)
                row[f"{metric}_current"] = a
                row[f"{metric}_historical"] = b
                row[f"{metric}_delta"] = a - b if a is not None and b is not None else None
                row[f"{metric}_change_pct"] = ((a - b) / abs(b) * 100) if a is not None and b not in (None, 0) else None
            comparisons.append(row)
    # Per-batch plan attainment, keeping unknown values as N/A.
    plan_attainment = []
    for b in batches:
        m = b["metrics"]
        for metric in ["leads", "applications", "enrollments", "gsv_cr", "spends"]:
            actual, plan = m.get(metric), m.get(f"plan_{metric}")
            plan_attainment.append({"batch": b["program_code"], "metric": metric, "actual": actual, "plan": plan,
                "variance": actual - plan if actual is not None and plan is not None else None,
                "variance_pct": ((actual - plan) / abs(plan) * 100) if actual is not None and plan not in (None, 0) else None})
    for b in batches:
        m = b["metrics"]
        for metric, plan_col in PAID_MAP.items():
            actual, plan = m.get(metric), m.get(plan_col)
            plan_attainment.append({"batch": b["program_code"], "metric": metric, "actual": actual, "plan": plan,
                "variance": actual - plan if actual is not None and plan is not None else None,
                "variance_pct": ((actual - plan) / abs(plan) * 100) if actual is not None and plan not in (None, 0) else None})
    # Planned vs actual trend is capped to the selected batch plus up to three ancestors.
    # Decompose GSV change into enrollment-volume and fee effects for adjacent available batches.
    gsv_impact = []
    for idx in range(len(batches) - 1):
        newer, older = batches[idx], batches[idx + 1]
        nm, om = newer["metrics"], older["metrics"]
        e_new, e_old = nm.get("enrollments"), om.get("enrollments")
        f_new, f_old = nm.get("program_fees"), om.get("program_fees")
        g_new, g_old = nm.get("gsv_cr"), om.get("gsv_cr")
        volume_effect = (e_new - e_old) * f_old / 10_000_000 if None not in (e_new, e_old, f_old) else None
        fee_effect = (f_new - f_old) * e_new / 10_000_000 if None not in (f_new, f_old, e_new) else None
        gsv_delta = g_new - g_old if None not in (g_new, g_old) else None
        expected_delta = volume_effect + fee_effect if volume_effect is not None and fee_effect is not None else None
        gsv_impact.append({
            "current_batch": newer["program_code"], "previous_batch": older["program_code"],
            "current_fees": f_new, "previous_fees": f_old,
            "fee_change": f_new - f_old if None not in (f_new, f_old) else None,
            "fee_change_pct": ((f_new - f_old) / abs(f_old) * 100) if None not in (f_new, f_old) and f_old != 0 else None,
            "current_enrollments": e_new, "previous_enrollments": e_old,
            "observed_gsv_change_cr": gsv_delta, "enrollment_volume_effect_cr": volume_effect,
            "fee_price_effect_cr": fee_effect, "decomposed_gsv_change_cr": expected_delta,
            "reconciliation_difference_cr": (gsv_delta - expected_delta) if gsv_delta is not None and expected_delta is not None else None,
            "gsv_reconciles_to_fee_times_enrollments": (abs(gsv_delta - expected_delta) < 0.000001) if gsv_delta is not None and expected_delta is not None else None,
        })
    warnings = []
    # Explicitly flag dangling previous_batch_code references, without treating them as real batches.
    available_codes = set(clean[PROGRAM_COL].dropna().astype(str).str.strip())
    for b in batches:
        raw_previous = b.get("previous_batch_code")
        if raw_previous and str(raw_previous).strip() not in available_codes:
            warnings.append(f"{b['program_code']} references previous batch {str(raw_previous).strip()}, but that program_code is not present in the dataset; it is not included as an analyzed batch.")
    if len(lineage) < 2:
        warnings.append("No historical ancestor batch was found in the available dataset.")
    for b in batches:
        m = b["metrics"]
        if m.get("spend_missing_rows", 0):
            warnings.append(f"{b['program_code']}: spend missing in {m['spend_missing_rows']} of {m['row_count']} source rows; spend-based metrics may be partial or unavailable.")
        if b["status"] and str(b["status"]).lower() not in {"closed", "complete", "completed"} and b["program_code"] != program_code:
            warnings.append(f"Historical batch {b['program_code']} has status {b['status']}; compare lifecycle carefully.")
    return {"program_code": program_code, "lineage_newest_first": lineage, "batches": batches, "comparisons_to_each_history": comparisons,
            "plan_attainment": plan_attainment, "gsv_price_impact": gsv_impact, "max_batches": max_batches, "full_lineage_count": len(full_lineage), "history_truncated": (selected_old_batches is None and len(full_lineage) > len(lineage)), "selection_mode": "user_selected" if selected_old_batches is not None else "automatic_lineage", "warnings": list(dict.fromkeys(warnings)),
            "definitions": {"CPL": "spends / leads", "cost_per_application": "spends / applications", "CAC": "spends / enrollments",
                            "conversion": "numerator / denominator * 100", "GSV_to_spend": "GSV / spend; only meaningful if spend coverage is valid"}}


def _fmt(v, pct=False, cr=False):
    if v is None or pd.isna(v): return "N/A"
    if pct: return f"{v:.2f}%"
    if cr: return f"₹{v:.4f} cr"
    return f"{v:,.2f}" if isinstance(v, (float, np.floating, int, np.integer)) else str(v)


def format_batch_analysis_markdown(result: dict[str, Any]) -> str:
    batches = result["batches"]
    if not batches: return "No batch data available."
    lines = [f"# Batch Analysis: {result['program_code']}", "", "## 1. Batch lineage and comparability", "", "Batch history (newest to oldest): " + " → ".join(result["lineage_newest_first"]), ""]
    if result.get("history_truncated"):
        lines.append(f"Showing the latest {len(batches)} batches (maximum configured: {result.get('max_batches', 4)}); older available history is not included in this report.")
        lines.append("")
    lines += ["| Batch in dataset | Previous-batch reference | Status | Sales start | Leads | Warm leads | Applications | Enrollments | GSV | Spend coverage |", "|---|---|---|---|---:|---:|---:|---:|---:|---|"]
    for b in batches:
        m=b['metrics']; lines.append(f"| {b['program_code']} | {b.get('previous_batch_display', 'N/A')} | {b['status'] or 'N/A'} | {str(b['sales_start_date'])[:10] if b['sales_start_date'] else 'N/A'} | {_fmt(m.get('leads'))} | {_fmt(m.get('warm_leads'))} | {_fmt(m.get('applications'))} | {_fmt(m.get('enrollments'))} | {_fmt(m.get('gsv_cr'),cr=True)} | {m.get('spend_coverage_rows',0)}/{m.get('row_count',0)} rows |")
    lines += ["", "## 2. All-metric batch trend", "", "| Metric | " + " | ".join(result["lineage_newest_first"]) + " |", "|---|" + "---:|"*len(batches)]
    trend_metrics = [("Program fees","program_fees",False,False),("Fee-implied GSV","fee_implied_gsv_cr",False,True),("Leads","leads",False,False),("Warm leads","warm_leads",False,False),("Applications","applications",False,False),("Enrollments","enrollments",False,False),("GSV","gsv_cr",False,True),("Spend","spends",False,False),("CPL","cpl",False,False),("Cost/application","cost_per_application",False,False),("CAC","cac",False,False),("Lead→warm conversion","lead_to_warm_pct",True,False),("Lead→application conversion","lead_to_application_pct",True,False),("Warm→application conversion","warm_to_application_pct",True,False),("Application→enrollment conversion","application_to_enrollment_pct",True,False),("Lead→enrollment conversion","lead_to_enrollment_pct",True,False),("GSV/spend","gsv_to_spend",False,False),("Paid leads","paid_leads",False,False),("Paid enrollments","paid_enrollments",False,False),("Paid GSV","paid_gsv_cr",False,True),("Paid spend","paid_spends",False,False),("Paid CPL","paid_cpl",False,False),("Paid CAC","paid_cac",False,False),("Paid GSV/spend","paid_gsv_to_spend",False,False)]
    for label,key,pct,cr in trend_metrics:
        lines.append("| " + label + " | " + " | ".join(_fmt(b['metrics'].get(key),pct=pct,cr=cr) for b in batches) + " |")
    lines += ["", "## 3. Planned data comparison across available batches (maximum four batches)", "", "| Batch | Metric | Actual | Plan | Variance | Variance % |", "|---|---|---:|---:|---:|---:|"]
    for r in result["plan_attainment"]:
        if r['actual'] is None and r['plan'] is None: continue
        lines.append(f"| {r['batch']} | {r['metric']} | {_fmt(r['actual'],cr=r['metric'] in {'gsv_cr','paid_gsv_cr','paid_gsv'})} | {_fmt(r['plan'],cr=r['metric'] in {'gsv_cr','paid_gsv_cr','paid_gsv'})} | {_fmt(r['variance'],cr=r['metric'] in {'gsv_cr','paid_gsv_cr','paid_gsv'})} | {_fmt(r['variance_pct'],pct=True)} |")
    lines += ["", "## 4. Current batch vs every historical batch", "", "| Historical batch | Metric | Current | Historical | Absolute change | Relative change |", "|---|---|---:|---:|---:|---:|"]
    for r in result['comparisons_to_each_history']:
        for metric in ["leads","warm_leads","applications","enrollments","gsv_cr","spends","cpl","cost_per_application","cac","lead_to_warm_pct","lead_to_application_pct","warm_to_application_pct","application_to_enrollment_pct","lead_to_enrollment_pct"]:
            pct = metric.endswith("_pct")
            cr = metric == "gsv_cr"
            lines.append(f"| {r['historical_batch']} | {metric} | {_fmt(r.get(metric+'_current'),pct=pct,cr=cr)} | {_fmt(r.get(metric+'_historical'),pct=pct,cr=cr)} | {_fmt(r.get(metric+'_delta'),pct=pct,cr=cr)} | {_fmt(r.get(metric+'_change_pct'),pct=True)} |")
    lines += ["", "## 5. Program fee (price point) and GSV impact", "", "Program fees are treated as a program-level value, not summed across source rows. The decomposition below estimates the change in GSV associated with enrollment volume versus fee changes, using prior fees for the volume effect and current enrollments for the fee effect. It is only interpretable where GSV reconciles to fees × enrollments.", "", "| Current batch | Previous batch | Previous fee | Current fee | Fee change | Enrollment change | Observed GSV change | Enrollment-volume effect | Fee/price effect | Reconciliation difference |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result.get("gsv_price_impact", []):
        lines.append(f"| {r['current_batch']} | {r['previous_batch']} | {_fmt(r.get('previous_fees'))} | {_fmt(r.get('current_fees'))} | {_fmt(r.get('fee_change'))} ({_fmt(r.get('fee_change_pct'), pct=True)}) | {_fmt((r.get('current_enrollments') - r.get('previous_enrollments')) if None not in (r.get('current_enrollments'), r.get('previous_enrollments')) else None)} | {_fmt(r.get('observed_gsv_change_cr'), cr=True)} | {_fmt(r.get('enrollment_volume_effect_cr'), cr=True)} | {_fmt(r.get('fee_price_effect_cr'), cr=True)} | {_fmt(r.get('reconciliation_difference_cr'), cr=True)} |")
        if r.get("gsv_reconciles_to_fee_times_enrollments") is False:
            lines.append(f"| Note | {r['current_batch']} vs {r['previous_batch']}: observed GSV does not fully reconcile to the fee × enrollment decomposition; interpret fee/volume attribution cautiously. | | | | | | | | |")
    if not result.get("gsv_price_impact"):
        lines.append("No adjacent batch pair is available for fee and GSV impact analysis.")
    lines += ["", "## 6. Hidden patterns and diagnostic signals", ""]
    if len(batches) >= 2:
        cur=batches[0]['metrics']; prev=batches[1]['metrics']
        for label, key, higher_better in [("Leads","leads",True),("Warm leads","warm_leads",True),("Applications","applications",True),("Enrollments","enrollments",True),("GSV","gsv_cr",True),("Spend","spends",None),("Lead-to-warm conversion","lead_to_warm_pct",True),("Lead-to-application conversion","lead_to_application_pct",True),("Warm-to-application conversion","warm_to_application_pct",True),("Application-to-enrollment conversion","application_to_enrollment_pct",True),("CPL","cpl",False),("CAC","cac",False)]:
            a,b=cur.get(key),prev.get(key)
            if a is None or b is None: continue
            delta=a-b
            if delta == 0: continue
            direction="increased" if delta>0 else "decreased"
            lines.append(f"- {label} {direction} from {_fmt(b,pct=key.endswith('_pct'))} to {_fmt(a,pct=key.endswith('_pct'))} ({_fmt(delta,pct=key.endswith('_pct'))} absolute change).")
        if cur.get('leads') and prev.get('leads') and cur.get('lead_to_application_pct') is not None and prev.get('lead_to_application_pct') is not None:
            lines.append("- Interpret volume and conversion separately: more leads do not necessarily imply stronger downstream outcomes; conversion rates are shown alongside volume for that reason.")
    else:
        lines.append("- Historical comparisons are unavailable because no prior batch was found.")
    lines += ["", "## 7. Data limitations", ""]
    lines.extend([f"- {w}" for w in result.get('warnings', [])] or ["- No major automated data warnings were detected; lifecycle comparability should still be reviewed before causal interpretation."])
    lines += ["", "## 8. Business conclusion and next investigations", "", "Use this trend report to identify where volume, conversion, economics or plan attainment shifted across the available batch history. Treat explanations as hypotheses until campaign, CRM, cohort-age and operational evidence confirms them. Prioritize the largest absolute outcome changes, then test whether they arise from source mix, within-source conversion, spend coverage or selling-window differences."]
    return "\n".join(lines)


batch_analysis = analyze_batch_history
