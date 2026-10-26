"""Deterministic source-level analysis for the Program Performance Agent.

This module is intentionally separate from the overall portfolio analysis and
is not wired into app.py. It analyzes one selected program, grouped by the
source-level column (normally ``source_created``), using the unified
Batch_wise_program_data.csv schema.

Public API
----------
analyze_sources(df, program_code, previous_batch_code=None)
source_analysis = analyze_sources

All calculations are deterministic Python/pandas calculations. Missing values
remain unavailable; they are not silently converted to zero for ratios.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


PROGRAM_COL = "program_code"
DEFAULT_SOURCE_COL = "source_created"

# These columns are additive at source level, when present.
ACTUAL_ADDITIVE = [
    "leads",
    "warm_leads",
    "applications",
    "enrollments",
    "gsv_cr",
    "spends",
]

PAID_ACTUAL_ADDITIVE = [
    "paid_leads",
    "paid_enrollments",
    "paid_gsv_cr",
    "paid_spends",
]

# Plan fields are additive if the file stores source-level plans. If a plan is
# repeated identically on every source row, we deduplicate it within each
# source before creating program-level totals.
PLAN_ADDITIVE = [
    "Planned_Budget",
    "Planned_Lead",
    "Planned_Application",
    "Planned_Enrollment",
    "Planned_GSV",
]

PAID_PLAN_ADDITIVE = [
    "planned_paid_leads",
    "planned_paid_enrollments",
    "planned_paid_gsv",
    "planned_paid_spends",
]

PREVIOUS_ACTUAL_MAP = {
    "leads": "previous_batch_leads",
    "warm_leads": "previous_batch_warm_leads",
    "applications": "previous_batch_applications",
    "enrollments": "previous_batch_enrollments",
    "gsv_cr": "previous_batch_gsv_cr",
    "spends": "previous_batch_spends",
}

PREVIOUS_PAID_MAP = {
    "paid_leads": "previous_batch_paid_leads",
    "paid_enrollments": "previous_batch_paid_enrollments",
    "paid_gsv_cr": "previous_batch_paid_gsv_cr",
    "paid_spends": "previous_batch_paid_spends",
}

PREVIOUS_PLAN_MAP = {
    "Planned_Budget": "previous_batch_Planned_Budget",
    "Planned_Lead": "previous_batch_Planned_Lead",
    "Planned_Application": "previous_batch_Planned_Application",
    "Planned_Enrollment": "previous_batch_Planned_Enrollment",
    "Planned_GSV": "previous_batch_Planned_GSV",
}

PREVIOUS_PAID_PLAN_MAP = {
    "planned_paid_leads": "previous_batch_planned_paid_leads",
    "planned_paid_enrollments": "previous_batch_planned_paid_enrollments",
    "planned_paid_gsv": "previous_batch_planned_paid_gsv",
    "planned_paid_spends": "previous_batch_planned_paid_spends",
}

# Fallbacks allow a future column rename without mixing source analysis into
# the overall analysis. ``source_created`` remains the primary expected name.
SOURCE_COLUMN_CANDIDATES = (
    "source_created",
    "source",
    "source_name",
    "source_level",
    "acquisition_source",
    "lead_source",
)


def _is_missing(value: Any) -> bool:
    try:
        return value is None or bool(pd.isna(value))
    except (TypeError, ValueError):
        return value is None


def _number(value: Any) -> Optional[float]:
    if _is_missing(value):
        return None
    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip()
            if not value:
                return None
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def _clean_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if column in _numeric_columns():
            out[column] = pd.to_numeric(
                out[column].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )
    return out


def _numeric_columns() -> set[str]:
    cols = set(
        ACTUAL_ADDITIVE
        + PAID_ACTUAL_ADDITIVE
        + PLAN_ADDITIVE
        + PAID_PLAN_ADDITIVE
        + list(PREVIOUS_ACTUAL_MAP.values())
        + list(PREVIOUS_PAID_MAP.values())
        + list(PREVIOUS_PLAN_MAP.values())
        + list(PREVIOUS_PAID_PLAN_MAP.values())
    )
    cols.update(["total_program_fees"])
    return cols


def _safe_sum(series: pd.Series) -> Optional[float]:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return None
    return float(numeric.sum(min_count=1))


def _safe_ratio(numerator: Any, denominator: Any, multiplier: float = 1.0) -> Optional[float]:
    n = _number(numerator)
    d = _number(denominator)
    if n is None or d is None or d == 0:
        return None
    return round((n / d) * multiplier, 4)


def _delta(current: Any, previous: Any) -> dict[str, Optional[float]]:
    c = _number(current)
    p = _number(previous)
    if c is None or p is None:
        return {"absolute": None, "percent": None}
    difference = c - p
    percent = None if p == 0 else (difference / abs(p)) * 100
    return {
        "absolute": round(difference, 4),
        "percent": round(percent, 2) if percent is not None else None,
    }


def _status(metric: str, current: Any, comparison: Any) -> str:
    c = _number(current)
    p = _number(comparison)
    if c is None or p is None:
        return "not_available"
    if np.isclose(c, p):
        return "flat"
    # Spend is context-dependent: a reduction is not automatically an
    # improvement, and an increase is not automatically deterioration.
    if metric in {"spends", "spend", "paid_spends"}:
        return "context_dependent"
    lower_is_better = metric in {
        "cpl", "cac", "cost_per_application", "cost_per_warm_lead",
        "paid_cpl", "paid_cac"
    }
    if lower_is_better:
        return "improved" if c < p else "deteriorated"
    return "improved" if c > p else "deteriorated"


def _detect_source_column(df: pd.DataFrame, source_column: Optional[str] = None) -> str:
    if source_column:
        if source_column not in df.columns:
            raise ValueError(
                f"Source column '{source_column}' was requested but is not in the data. "
                f"Available columns: {', '.join(map(str, df.columns))}"
            )
        return source_column
    for candidate in SOURCE_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Could not identify the source-level column. Expected 'source_created' "
        "or one of: " + ", ".join(SOURCE_COLUMN_CANDIDATES[1:])
    )


def _scope_program(df: pd.DataFrame, program_code: str) -> pd.DataFrame:
    if PROGRAM_COL not in df.columns:
        raise ValueError(f"Required column '{PROGRAM_COL}' is missing.")
    target = str(program_code).strip().casefold()
    scoped = df[
        df[PROGRAM_COL].astype(str).str.strip().str.casefold().eq(target)
    ].copy()
    if scoped.empty:
        raise ValueError(f"No rows found for program '{program_code}'.")
    return scoped


def _first_non_null(series: pd.Series) -> Optional[float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else None


def _source_metric_value(group: pd.DataFrame, column: str) -> Optional[float]:
    """Sum a source-level metric only when every row in scope is populated.

    A partial sum would understate a source total while looking complete, so
    missing values make the aggregate unavailable rather than silently zero.
    """
    if column not in group.columns:
        return None
    values = pd.to_numeric(group[column], errors="coerce")
    if values.empty or values.notna().sum() == 0 or values.isna().any():
        return None
    return float(values.sum())


def _unique_program_value(series: pd.Series) -> Optional[float]:
    """Read a program-level value repeated on source rows exactly once.

    If repeated values conflict, return None rather than summing or arbitrarily
    choosing one. The caller separately reports the inconsistency.
    """
    values = pd.to_numeric(series, errors="coerce").dropna().drop_duplicates()
    if values.empty or len(values) != 1:
        return None
    return float(values.iloc[0])


def _program_metric_values(frame: pd.DataFrame, columns: list[str]) -> dict[str, Optional[float]]:
    return {
        column: _unique_program_value(frame[column]) if column in frame.columns else None
        for column in columns
    }


def _source_plan_value(group: pd.DataFrame, column: str) -> Optional[float]:
    """Read source plan safely, avoiding repeated identical plan values.

    If the plan differs by row within the source, sum the distinct non-null
    values as a best-effort interpretation of source-level plan records.
    """
    if column not in group.columns:
        return None
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    if values.empty:
        return None
    distinct = values.drop_duplicates()
    if len(distinct) == 1:
        return float(distinct.iloc[0])
    return float(values.sum())


def _metric_bundle(
    group: pd.DataFrame,
    actual_map: dict[str, str],
    plan_map: Optional[dict[str, str]] = None,
    paid: bool = False,
) -> dict[str, Any]:
    actual = {}
    for metric in actual_map:
        actual[metric] = _source_metric_value(group, actual_map[metric])

    if paid:
        leads = actual.get("paid_leads")
        enrollments = actual.get("paid_enrollments")
        spend = actual.get("paid_spends")
        gsv = actual.get("paid_gsv_cr")
        actual.update({
            "paid_cpl": _safe_ratio(spend, leads),
            "paid_cac": _safe_ratio(spend, enrollments),
            "paid_gsv_per_enrollment_cr": _safe_ratio(gsv, enrollments),
        })
    else:
        leads = actual.get("leads")
        warm_leads = actual.get("warm_leads")
        applications = actual.get("applications")
        enrollments = actual.get("enrollments")
        spend = actual.get("spends")
        gsv = actual.get("gsv_cr")
        actual.update({
            "cpl": _safe_ratio(spend, leads),
            "cost_per_warm_lead": _safe_ratio(spend, warm_leads),
            "cost_per_application": _safe_ratio(spend, applications),
            "cac": _safe_ratio(spend, enrollments),
            "lead_to_warm_lead_pct": _safe_ratio(warm_leads, leads, 100),
            "lead_to_application_pct": _safe_ratio(applications, leads, 100),
            "warm_lead_to_application_pct": _safe_ratio(applications, warm_leads, 100),
            "application_to_enrollment_pct": _safe_ratio(enrollments, applications, 100),
            "lead_to_enrollment_pct": _safe_ratio(enrollments, leads, 100),
            "gsv_per_enrollment_cr": _safe_ratio(gsv, enrollments),
            "spend_share_basis": spend,
        })

    plan = {}
    if plan_map:
        for metric, column in plan_map.items():
            plan[metric] = _source_plan_value(group, column)

    return {"actual": actual, "plan": plan}


def _make_source_summary(
    source_name: str,
    group: pd.DataFrame,
    current_bundle: dict[str, Any],
    previous_bundle: dict[str, Any],
    row_count: int,
) -> dict[str, Any]:
    current = current_bundle["actual"]
    plan = current_bundle["plan"]
    previous = previous_bundle["actual"]
    previous_plan = previous_bundle["plan"]

    # Current versus previous-batch metric deltas.
    comparison_metrics = [
        "leads", "warm_leads", "applications", "enrollments", "gsv_cr", "spends",
        "cpl", "cost_per_warm_lead", "cost_per_application", "cac",
        "lead_to_warm_lead_pct", "lead_to_application_pct",
        "warm_lead_to_application_pct", "application_to_enrollment_pct",
        "lead_to_enrollment_pct",
    ]
    comparisons = {}
    for metric in comparison_metrics:
        c = current.get(metric)
        p = previous.get(metric)
        comparisons[metric] = {
            **_delta(c, p),
            "status": _status(metric, c, p),
            "current": c,
            "previous": p,
            "unit": "percentage_points" if metric.endswith("_pct") else "value",
        }
        if metric.endswith("_pct") and c is not None and p is not None:
            comparisons[metric]["percentage_point_change"] = round(c - p, 4)

    plan_comparisons = {}
    plan_metric_map = {
        "leads": "Planned_Lead",
        "applications": "Planned_Application",
        "enrollments": "Planned_Enrollment",
        "gsv_cr": "Planned_GSV",
        "spends": "Planned_Budget",
    }
    for metric, plan_key in plan_metric_map.items():
        actual_value = current.get(metric)
        plan_value = plan.get(plan_key)
        variance = _delta(actual_value, plan_value)
        plan_comparisons[metric] = {
            "actual": actual_value,
            "plan": plan_value,
            "variance_absolute": variance["absolute"],
            "variance_percent": variance["percent"],
            "status": _status(metric, actual_value, plan_value),
        }

    paid_actual = current_bundle.get("paid_actual", {})
    paid_plan = current_bundle.get("paid_plan", {})

    return {
        "source": source_name,
        "rows": int(row_count),
        "metadata": {
            col: (group[col].dropna().astype(str).iloc[0] if col in group.columns and group[col].notna().any() else None)
            for col in ["category", "tech_management", "New_Repeat", "financial_year", "program_status_f"]
        },
        "current": current,
        "plan": plan,
        "previous": previous,
        "previous_plan": previous_plan,
        "paid_current": paid_actual,
        "paid_plan": paid_plan,
        "comparison_to_previous": comparisons,
        "comparison_to_plan": plan_comparisons,
    }


def analyze_sources(
    df: pd.DataFrame,
    program_code: str,
    previous_batch_code: Optional[str] = None,
    source_column: Optional[str] = None,
) -> dict[str, Any]:
    """Return detailed source-level performance for one selected program.

    Parameters
    ----------
    df:
        Unified batch/program/source dataframe.
    program_code:
        Selected current program code.
    previous_batch_code:
        Optional label for the prior batch. The supplied CSV's
        ``previous_batch_*`` columns are used for prior-period values. This
        argument labels the comparison only; it does not fetch another batch.
    source_column:
        Optional explicit source dimension. Defaults to ``source_created``.

    Returns
    -------
    dict
        Includes program scope, totals, per-source metrics, source shares,
        current-vs-previous comparison, plan comparison, key observations,
        missing-data notes, and a compact table-ready DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("The supplied dataset is empty.")
    if program_code is None or not str(program_code).strip():
        raise ValueError("program_code is required.")

    source_col = _detect_source_column(df, source_column)
    scoped = _clean_numeric_columns(_scope_program(df, program_code))

    # Blank sources are retained as a visible bucket rather than discarded.
    scoped[source_col] = scoped[source_col].where(
        scoped[source_col].notna(), "(Source not recorded)"
    )
    scoped[source_col] = scoped[source_col].astype(str).str.strip().replace(
        {"": "(Source not recorded)", "nan": "(Source not recorded)", "None": "(Source not recorded)"}
    )

    if previous_batch_code is None and "previous_batch_code" in scoped.columns:
        previous_values = scoped["previous_batch_code"].dropna().astype(str).str.strip()
        previous_values = previous_values[previous_values.ne("")]
        if not previous_values.empty:
            previous_batch_code = previous_values.iloc[0]

    source_groups = list(scoped.groupby(source_col, dropna=False, sort=False))
    if not source_groups:
        raise ValueError(f"No source rows found for program '{program_code}'.")

    source_results = []
    for source_name, group in source_groups:
        current_actual = {m: _source_metric_value(group, m) for m in ACTUAL_ADDITIVE}
        current_actual.update({
            "cpl": _safe_ratio(current_actual.get("spends"), current_actual.get("leads")),
            "cost_per_warm_lead": _safe_ratio(current_actual.get("spends"), current_actual.get("warm_leads")),
            "cost_per_application": _safe_ratio(current_actual.get("spends"), current_actual.get("applications")),
            "cac": _safe_ratio(current_actual.get("spends"), current_actual.get("enrollments")),
            "lead_to_warm_lead_pct": _safe_ratio(current_actual.get("warm_leads"), current_actual.get("leads"), 100),
            "lead_to_application_pct": _safe_ratio(current_actual.get("applications"), current_actual.get("leads"), 100),
            "warm_lead_to_application_pct": _safe_ratio(current_actual.get("applications"), current_actual.get("warm_leads"), 100),
            "application_to_enrollment_pct": _safe_ratio(current_actual.get("enrollments"), current_actual.get("applications"), 100),
            "lead_to_enrollment_pct": _safe_ratio(current_actual.get("enrollments"), current_actual.get("leads"), 100),
            "gsv_per_enrollment_cr": _safe_ratio(current_actual.get("gsv_cr"), current_actual.get("enrollments")),
        })

        current_plan = {m: _source_plan_value(group, m) for m in PLAN_ADDITIVE}
        # Paid metrics and paid plans are program-level fields repeated on
        # each source row. Do not assign the full program total to every source.
        paid_actual = {m: None for m in PAID_ACTUAL_ADDITIVE}
        paid_actual.update({"paid_cpl": None, "paid_cac": None, "paid_gsv_per_enrollment_cr": None})
        paid_plan = {m: None for m in PAID_PLAN_ADDITIVE}

        previous_actual = {}
        for metric, column in PREVIOUS_ACTUAL_MAP.items():
            previous_actual[metric] = _source_metric_value(group, column)
        previous_actual.update({
            "cpl": _safe_ratio(previous_actual.get("spends"), previous_actual.get("leads")),
            "cost_per_warm_lead": _safe_ratio(previous_actual.get("spends"), previous_actual.get("warm_leads")),
            "cost_per_application": _safe_ratio(previous_actual.get("spends"), previous_actual.get("applications")),
            "cac": _safe_ratio(previous_actual.get("spends"), previous_actual.get("enrollments")),
            "lead_to_warm_lead_pct": _safe_ratio(previous_actual.get("warm_leads"), previous_actual.get("leads"), 100),
            "lead_to_application_pct": _safe_ratio(previous_actual.get("applications"), previous_actual.get("leads"), 100),
            "warm_lead_to_application_pct": _safe_ratio(previous_actual.get("applications"), previous_actual.get("warm_leads"), 100),
            "application_to_enrollment_pct": _safe_ratio(previous_actual.get("enrollments"), previous_actual.get("applications"), 100),
            "lead_to_enrollment_pct": _safe_ratio(previous_actual.get("enrollments"), previous_actual.get("leads"), 100),
            "gsv_per_enrollment_cr": _safe_ratio(previous_actual.get("gsv_cr"), previous_actual.get("enrollments")),
        })

        previous_plan = {m: _source_plan_value(group, col) for m, col in PREVIOUS_PLAN_MAP.items()}
        # Historical paid metrics/plans are likewise repeated program-level
        # values, not source-level benchmarks.
        previous_paid_actual = {m: None for m in PREVIOUS_PAID_MAP}
        previous_paid_actual.update({"paid_cpl": None, "paid_cac": None, "paid_gsv_per_enrollment_cr": None})
        previous_paid_plan = {m: None for m in PREVIOUS_PAID_PLAN_MAP}

        comparisons = {}
        metric_pairs = [
            ("leads", "leads"), ("warm_leads", "warm_leads"),
            ("applications", "applications"), ("enrollments", "enrollments"),
            ("gsv_cr", "gsv_cr"), ("spends", "spends"),
            ("cpl", "cpl"), ("cac", "cac"),
            ("cost_per_warm_lead", "cost_per_warm_lead"),
            ("cost_per_application", "cost_per_application"),
            ("lead_to_warm_lead_pct", "lead_to_warm_lead_pct"),
            ("lead_to_application_pct", "lead_to_application_pct"),
            ("warm_lead_to_application_pct", "warm_lead_to_application_pct"),
            ("application_to_enrollment_pct", "application_to_enrollment_pct"),
            ("lead_to_enrollment_pct", "lead_to_enrollment_pct"),
        ]
        for metric, key in metric_pairs:
            c = current_actual.get(key)
            p = previous_actual.get(key)
            item = {"current": c, "previous": p, **_delta(c, p), "status": _status(key, c, p)}
            if key.endswith("_pct") and c is not None and p is not None:
                item["percentage_point_change"] = round(c - p, 4)
                item["unit"] = "percentage_points"
            else:
                item["unit"] = "value"
            comparisons[metric] = item

        plan_comparison = {}
        plan_links = {
            "leads": (current_actual.get("leads"), current_plan.get("Planned_Lead")),
            "applications": (current_actual.get("applications"), current_plan.get("Planned_Application")),
            "enrollments": (current_actual.get("enrollments"), current_plan.get("Planned_Enrollment")),
            "gsv_cr": (current_actual.get("gsv_cr"), current_plan.get("Planned_GSV")),
            "spends": (current_actual.get("spends"), current_plan.get("Planned_Budget")),
        }
        for metric, (actual_value, plan_value) in plan_links.items():
            variance = _delta(actual_value, plan_value)
            plan_comparison[metric] = {
                "actual": actual_value,
                "plan": plan_value,
                "variance_absolute": variance["absolute"],
                "variance_percent": variance["percent"],
                "status": _status(metric, actual_value, plan_value),
            }

        source_results.append({
            "source": str(source_name),
            "row_count": int(len(group)),
            "current": current_actual,
            "plan": current_plan,
            "paid_current": paid_actual,
            "paid_plan": paid_plan,
            "previous": previous_actual,
            "previous_plan": previous_plan,
            "previous_paid": previous_paid_actual,
            "previous_paid_plan": previous_paid_plan,
            "comparison_to_previous": comparisons,
            "comparison_to_plan": plan_comparison,
            "metadata": {
                column: (group[column].dropna().astype(str).iloc[0] if column in group.columns and group[column].notna().any() else None)
                for column in ["category", "tech_management", "New_Repeat", "financial_year", "program_status_f"]
            },
        })

    # Totals are recalculated from raw selected-program rows, not by averaging
    # source CPL/CAC. Ratios therefore remain mathematically valid.
    totals = {m: _safe_sum(scoped[m]) if m in scoped.columns else None for m in ACTUAL_ADDITIVE}
    # Do not report aggregate spend as complete when one or more source rows
    # have missing spend. Known spend is exposed separately for reconciliation.
    known_spend = _safe_sum(scoped["spends"]) if "spends" in scoped.columns else None
    spend_incomplete = "spends" in scoped.columns and scoped["spends"].isna().any()
    if spend_incomplete:
        totals["spends"] = None
    totals.update({
        "cpl": _safe_ratio(totals.get("spends"), totals.get("leads")),
        "cost_per_warm_lead": _safe_ratio(totals.get("spends"), totals.get("warm_leads")),
        "cost_per_application": _safe_ratio(totals.get("spends"), totals.get("applications")),
        "cac": _safe_ratio(totals.get("spends"), totals.get("enrollments")),
        "lead_to_warm_lead_pct": _safe_ratio(totals.get("warm_leads"), totals.get("leads"), 100),
        "lead_to_application_pct": _safe_ratio(totals.get("applications"), totals.get("leads"), 100),
        "warm_lead_to_application_pct": _safe_ratio(totals.get("applications"), totals.get("warm_leads"), 100),
        "application_to_enrollment_pct": _safe_ratio(totals.get("enrollments"), totals.get("applications"), 100),
        "lead_to_enrollment_pct": _safe_ratio(totals.get("enrollments"), totals.get("leads"), 100),
        "gsv_per_enrollment_cr": _safe_ratio(totals.get("gsv_cr"), totals.get("enrollments")),
    })

    previous_totals = {m: _safe_sum(scoped[col]) if col in scoped.columns else None for m, col in PREVIOUS_ACTUAL_MAP.items()}
    previous_known_spend = _safe_sum(scoped["previous_batch_spends"]) if "previous_batch_spends" in scoped.columns else None
    previous_spend_incomplete = "previous_batch_spends" in scoped.columns and scoped["previous_batch_spends"].isna().any()
    if previous_spend_incomplete:
        previous_totals["spends"] = None
    previous_totals.update({
        "cpl": _safe_ratio(previous_totals.get("spends"), previous_totals.get("leads")),
        "cost_per_warm_lead": _safe_ratio(previous_totals.get("spends"), previous_totals.get("warm_leads")),
        "cost_per_application": _safe_ratio(previous_totals.get("spends"), previous_totals.get("applications")),
        "cac": _safe_ratio(previous_totals.get("spends"), previous_totals.get("enrollments")),
        "lead_to_warm_lead_pct": _safe_ratio(previous_totals.get("warm_leads"), previous_totals.get("leads"), 100),
        "lead_to_application_pct": _safe_ratio(previous_totals.get("applications"), previous_totals.get("leads"), 100),
        "warm_lead_to_application_pct": _safe_ratio(previous_totals.get("applications"), previous_totals.get("warm_leads"), 100),
        "application_to_enrollment_pct": _safe_ratio(previous_totals.get("enrollments"), previous_totals.get("applications"), 100),
        "lead_to_enrollment_pct": _safe_ratio(previous_totals.get("enrollments"), previous_totals.get("leads"), 100),
        "gsv_per_enrollment_cr": _safe_ratio(previous_totals.get("gsv_cr"), previous_totals.get("enrollments")),
    })

    # Source contribution shares use additive totals only. Efficiency metrics
    # are intentionally never summed or averaged to determine total efficiency.
    for result in source_results:
        result["share_of_program"] = {}
        for metric in ACTUAL_ADDITIVE:
            current_value = result["current"].get(metric)
            overall_value = totals.get(metric)
            result["share_of_program"][metric] = _safe_ratio(current_value, overall_value, 100)
        # This is explicitly share of recorded spend, not total program spend,
        # because source spend may be missing for one or more source buckets.
        result["recorded_spend_share_pct"] = _safe_ratio(
            result["current"].get("spends"), known_spend, 100
        )
        # Paid outcomes are program-level and cannot be attributed to an
        # individual source from these repeated fields.
        for metric in PAID_ACTUAL_ADDITIVE:
            result["share_of_program"][metric] = None

    # Paid totals and plan totals.
    paid_totals = _program_metric_values(scoped, PAID_ACTUAL_ADDITIVE)
    paid_totals.update({
        "paid_cpl": _safe_ratio(paid_totals.get("paid_spends"), paid_totals.get("paid_leads")),
        "paid_cac": _safe_ratio(paid_totals.get("paid_spends"), paid_totals.get("paid_enrollments")),
        "paid_gsv_per_enrollment_cr": _safe_ratio(paid_totals.get("paid_gsv_cr"), paid_totals.get("paid_enrollments")),
    })
    previous_paid_totals = {
        metric: _unique_program_value(scoped[column]) if column in scoped.columns else None
        for metric, column in PREVIOUS_PAID_MAP.items()
    }
    previous_paid_totals.update({
        "paid_cpl": _safe_ratio(previous_paid_totals.get("paid_spends"), previous_paid_totals.get("paid_leads")),
        "paid_cac": _safe_ratio(previous_paid_totals.get("paid_spends"), previous_paid_totals.get("paid_enrollments")),
        "paid_gsv_per_enrollment_cr": _safe_ratio(previous_paid_totals.get("paid_gsv_cr"), previous_paid_totals.get("paid_enrollments")),
    })
    paid_plan_totals = _program_metric_values(scoped, PAID_PLAN_ADDITIVE)
    previous_paid_plan_totals = {
        metric: _unique_program_value(scoped[column]) if column in scoped.columns else None
        for metric, column in PREVIOUS_PAID_PLAN_MAP.items()
    }

    # Paid vs non-paid is computed only for metrics with a validated paid
    # subset and a matching all-program total. Non-paid = total - paid.
    # Spend is unavailable for the all-program segment if any source spend is
    # missing, so non-paid spend efficiency is not fabricated.
    def _segment_values(total_values, paid_values, spend_is_incomplete=False):
        out = {}
        pairs = {
            "leads": ("leads", "paid_leads"),
            "enrollments": ("enrollments", "paid_enrollments"),
            "gsv_cr": ("gsv_cr", "paid_gsv_cr"),
            "spends": ("spends", "paid_spends"),
        }
        for metric, (total_key, paid_key) in pairs.items():
            total_value = _number(total_values.get(total_key))
            paid_value = _number(paid_values.get(paid_key))
            if metric == "spends" and spend_is_incomplete:
                out[metric] = None
            elif total_value is None or paid_value is None:
                out[metric] = None
            else:
                difference = total_value - paid_value
                out[metric] = round(difference, 4) if difference >= -1e-8 else None
        out["cpl"] = _safe_ratio(out.get("spends"), out.get("leads"))
        out["cac"] = _safe_ratio(out.get("spends"), out.get("enrollments"))
        out["gsv_to_spend"] = _safe_ratio(out.get("gsv_cr"), out.get("spends"))
        return out

    paid_vs_non_paid = {
        "current": {
            "paid": {
                **paid_totals,
                "gsv_to_spend": _safe_ratio((_number(paid_totals.get("paid_gsv_cr")) * 10_000_000) if _number(paid_totals.get("paid_gsv_cr")) is not None else None, paid_totals.get("paid_spends")),
            },
            "non_paid": _segment_values(totals, paid_totals, spend_incomplete),
            "total_program": totals,
        },
        "previous": {
            "paid": {
                **previous_paid_totals,
                "gsv_to_spend": _safe_ratio((_number(previous_paid_totals.get("paid_gsv_cr")) * 10_000_000) if _number(previous_paid_totals.get("paid_gsv_cr")) is not None else None, previous_paid_totals.get("paid_spends")),
            },
            "non_paid": _segment_values(previous_totals, previous_paid_totals, previous_spend_incomplete),
            "total_program": previous_totals,
        },
        "plan": {
            "paid": paid_plan_totals,
            "non_paid": {
                "leads": None,
                "enrollments": None,
                "gsv_cr": None,
                "spends": None,
            },
        },
        "limitations": [
            "Paid-vs-non-paid split is only available for leads, enrollments, GSV and spend because matching paid fields are supplied only for these metrics.",
            "Non-paid actuals are derived as all-program totals minus paid totals; negative or inconsistent differences are reported as unavailable.",
            "Non-paid plan values are unavailable because the file does not provide a separately identifiable non-paid plan; equal total and paid plan values are not assumed to mean a confirmed zero non-paid target.",
            "Non-paid spend and spend-based ratios remain unavailable when any source-level spend is missing.",
        ],
    }

    # Rank by enrollment contribution, then by enrollment volume.
    source_results.sort(
        key=lambda item: (
            _number(item["current"].get("enrollments")) or 0,
            _number(item["current"].get("leads")) or 0,
        ),
        reverse=True,
    )

    observations = _build_observations(source_results, totals, previous_totals)
    missing_metrics = [
        metric for metric in ACTUAL_ADDITIVE + PAID_ACTUAL_ADDITIVE
        if metric not in scoped.columns or scoped[metric].notna().sum() == 0
    ]
    if "Planned_Data_Status" in scoped.columns:
        statuses = scoped["Planned_Data_Status"].dropna().astype(str).str.strip().unique().tolist()
    else:
        statuses = []

    program_level_columns = (
        PAID_ACTUAL_ADDITIVE + PAID_PLAN_ADDITIVE
        + list(PREVIOUS_PAID_MAP.values()) + list(PREVIOUS_PAID_PLAN_MAP.values())
    )
    repeated_program_fields = []
    inconsistent_program_fields = []
    for column in program_level_columns:
        if column not in scoped.columns:
            continue
        values = pd.to_numeric(scoped[column], errors="coerce").dropna().drop_duplicates()
        if len(values) == 1 and len(scoped) > 1:
            repeated_program_fields.append(column)
        elif len(values) > 1:
            inconsistent_program_fields.append(column)

    data_quality = {
        "program_rows": int(len(scoped)),
        "distinct_sources": int(scoped[source_col].nunique(dropna=False)),
        "duplicate_full_rows": int(scoped.duplicated().sum()),
        "missing_source_labels_replaced": int(scoped[source_col].astype(str).eq("(Source not recorded)").sum()),
        "current_spend_missing_rows": int(scoped["spends"].isna().sum()) if "spends" in scoped.columns else int(len(scoped)),
        "previous_spend_missing_rows": int(scoped["previous_batch_spends"].isna().sum()) if "previous_batch_spends" in scoped.columns else int(len(scoped)),
        "repeated_program_level_fields_deduplicated": repeated_program_fields,
        "inconsistent_program_level_fields": inconsistent_program_fields,
        "planned_data_status_values": statuses,
    }

    # Build a source comparison table convenient for Streamlit tables or CSV export.
    table_rows = []
    for item in source_results:
        row = {"source": item["source"]}
        for key in [
            "leads", "warm_leads", "applications", "enrollments", "gsv_cr", "spends",
            "cpl", "cost_per_warm_lead", "cost_per_application", "cac",
            "lead_to_application_pct", "application_to_enrollment_pct", "lead_to_enrollment_pct",
        ]:
            row[key] = item["current"].get(key)
        row["enrollment_share_pct"] = item["share_of_program"].get("enrollments")
        row["spend_share_pct"] = item["share_of_program"].get("spends")
        row["recorded_spend_share_pct"] = item.get("recorded_spend_share_pct")
        row["paid_leads"] = None
        row["paid_enrollments"] = None
        row["paid_spends"] = None
        row["paid_cpl"] = None
        row["paid_cac"] = None
        row["previous_enrollments"] = item["previous"].get("enrollments")
        row["previous_cac"] = item["previous"].get("cac")
        row["enrollments_change_pct"] = item["comparison_to_previous"]["enrollments"]["percent"]
        row["cac_change_pct"] = item["comparison_to_previous"]["cac"]["percent"]
        table_rows.append(row)

    table = pd.DataFrame(table_rows)
    if not table.empty:
        table = table.replace({np.nan: None})

    return {
        "program_code": str(program_code),
        "source_column": source_col,
        "previous_batch_code": previous_batch_code,
        "row_count": int(len(scoped)),
        "source_count": int(len(source_results)),
        "sources": source_results,
        "totals": totals,
        "previous_totals": previous_totals,
        "paid_totals": paid_totals,
        "previous_paid_totals": previous_paid_totals,
        "paid_plan_totals": paid_plan_totals,
        "previous_paid_plan_totals": previous_paid_plan_totals,
        "paid_vs_non_paid": paid_vs_non_paid,
        "known_source_spend_sum": known_spend,
        "previous_known_source_spend_sum": previous_known_spend,
        "source_spend_incomplete": bool(spend_incomplete),
        "previous_source_spend_incomplete": bool(previous_spend_incomplete),
        "observations": observations,
        "missing_metrics": missing_metrics,
        "planned_data_status": statuses,
        "data_quality": data_quality,
        "table": table,
        "notes": [
            "CPL, CAC and conversion rates are calculated from aggregate numerators and denominators; source-level efficiency metrics are not averaged.",
            "Conversion movement is expressed in percentage points.",
            "Missing spend, GSV, paid or plan values are unavailable, not zero.",
            "Program-level paid metrics and paid plans repeated on source rows are deduplicated at program scope and are not attributed to individual sources.",
            "Aggregate spend-derived program ratios are unavailable if spend is missing for any source; known spend is reported separately as a partial sum.",
            "The previous_batch_* fields are used as supplied by the unified CSV; this module does not fetch another batch from a separate file.",
            "Source attribution describes observed associations and does not establish causality.",
        ],
    }


def _build_observations(
    sources: list[dict[str, Any]],
    totals: dict[str, Any],
    previous_totals: dict[str, Any],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    if not sources:
        return observations

    ranked_enrollment = sorted(
        sources,
        key=lambda x: _number(x["current"].get("enrollments")) or 0,
        reverse=True,
    )
    ranked_spend = sorted(
        [x for x in sources if _number(x["current"].get("spends")) is not None],
        key=lambda x: _number(x["current"].get("spends")) or 0,
        reverse=True,
    )
    ranked_cac = sorted(
        [x for x in sources if _number(x["current"].get("cac")) is not None],
        key=lambda x: _number(x["current"].get("cac")) or 0,
        reverse=True,
    )

    if ranked_enrollment:
        top = ranked_enrollment[0]
        value = _number(top["current"].get("enrollments"))
        share = _number(top.get("share_of_program", {}).get("enrollments"))
        if value is not None:
            observations.append({
                "type": "volume_leader",
                "severity": "info",
                "source": top["source"],
                "metric": "enrollments",
                "value": value,
                "share_pct": share,
                "message": (
                    f"{top['source']} has the highest enrollment volume "
                    f"({value:,.0f} enrollments; {share:.1f}% of program enrollments)."
                    if share is not None else
                    f"{top['source']} has the highest enrollment volume ({value:,.0f})."
                ),
            })

    if ranked_spend:
        top_spend = ranked_spend[0]
        spend = _number(top_spend["current"].get("spends"))
        spend_share = _number(top_spend.get("share_of_program", {}).get("spends"))
        cac = _number(top_spend["current"].get("cac"))
        enrollments = _number(top_spend["current"].get("enrollments"))
        observations.append({
            "type": "spend_leader",
            "severity": "info",
            "source": top_spend["source"],
            "metric": "spends",
            "value": spend,
            "share_pct": spend_share,
            "message": (
                f"{top_spend['source']} accounts for the largest spend "
                f"({spend:,.2f}; {spend_share:.1f}% of program spend)."
                if spend is not None and spend_share is not None else
                f"{top_spend['source']} has the largest recorded spend."
            ),
        })
        if spend is not None and spend > 0 and enrollments == 0:
            observations.append({
                "type": "no_enrollments_with_spend",
                "severity": "warning",
                "source": top_spend["source"],
                "metric": "enrollments",
                "value": 0,
                "message": f"{top_spend['source']} has recorded spend but no enrollments; CAC cannot be calculated.",
            })

    if ranked_cac:
        high = ranked_cac[0]
        high_cac = _number(high["current"].get("cac"))
        observations.append({
            "type": "highest_cac",
            "severity": "warning",
            "source": high["source"],
            "metric": "cac",
            "value": high_cac,
            "message": f"{high['source']} has the highest calculated CAC ({high_cac:,.2f}); review it alongside enrollment volume and spend share.",
        })

    # Sources where enrollments fell while CAC increased vs supplied previous data.
    for source in sources:
        comp = source.get("comparison_to_previous", {})
        enrollment_change = comp.get("enrollments", {})
        cac_change = comp.get("cac", {})
        enroll_pct = _number(enrollment_change.get("percent"))
        cac_pct = _number(cac_change.get("percent"))
        if enroll_pct is not None and cac_pct is not None and enroll_pct < 0 and cac_pct > 0:
            observations.append({
                "type": "volume_down_cac_up",
                "severity": "warning",
                "source": source["source"],
                "metric": "enrollments_and_cac",
                "enrollment_change_pct": enroll_pct,
                "cac_change_pct": cac_pct,
                "message": (
                    f"{source['source']} has lower enrollments ({enroll_pct:+.1f}%) "
                    f"and higher CAC ({cac_pct:+.1f}%) versus the supplied previous-batch figures."
                ),
            })

    # Conversion pressure: find the lowest non-null lead-to-enrollment rate.
    conversion_sources = [
        s for s in sources
        if _number(s["current"].get("lead_to_enrollment_pct")) is not None
    ]
    if conversion_sources:
        weakest = min(
            conversion_sources,
            key=lambda x: _number(x["current"].get("lead_to_enrollment_pct")) or 0,
        )
        rate = _number(weakest["current"].get("lead_to_enrollment_pct"))
        observations.append({
            "type": "lowest_funnel_conversion",
            "severity": "warning",
            "source": weakest["source"],
            "metric": "lead_to_enrollment_pct",
            "value": rate,
            "message": f"{weakest['source']} has the lowest lead-to-enrollment conversion ({rate:.2f}%); inspect its funnel volumes before attributing the cause.",
        })

    return observations


# Backward-friendly alias for callers that prefer this name.
source_analysis = analyze_sources


def format_source_analysis_markdown(result: dict[str, Any]) -> str:
    """Format the deterministic result as a compact, readable text report.

    This formatter contains no LLM calls; it is suitable as context for the
    existing agent or for displaying a report directly in Streamlit.
    """
    lines = [
        f"# Source-wise analysis: {result.get('program_code', 'Selected program')}",
        f"Source dimension: `{result.get('source_column')}`",
        f"Sources analyzed: {result.get('source_count', 0)}",
    ]
    if result.get("previous_batch_code"):
        lines.append(f"Previous batch comparison: `{result['previous_batch_code']}`")

    totals = result.get("totals", {})
    lines.extend([
        "",
        "## Program totals across sources",
        f"- Leads: {_fmt(totals.get('leads'))}",
        f"- Warm leads: {_fmt(totals.get('warm_leads'))}",
        f"- Applications: {_fmt(totals.get('applications'))}",
        f"- Enrollments: {_fmt(totals.get('enrollments'))}",
        f"- Spend: {_fmt(totals.get('spends'))}",
        f"- CPL: {_fmt(totals.get('cpl'))}",
        f"- CAC: {_fmt(totals.get('cac'))}",
        f"- GSV (Cr): {_fmt(totals.get('gsv_cr'))}",
        f"- Lead → Application: {_fmt(totals.get('lead_to_application_pct'), suffix='%')}",
        f"- Application → Enrollment: {_fmt(totals.get('application_to_enrollment_pct'), suffix='%')}",
        f"- Lead → Enrollment: {_fmt(totals.get('lead_to_enrollment_pct'), suffix='%')}",
        "",
        "## Source observations",
    ])
    observations = result.get("observations", [])
    if observations:
        lines.extend(f"- {item.get('message', '')}" for item in observations)
    else:
        lines.append("- No source-level observations could be generated from the available metrics.")

    lines.extend([
        "",
        "## Source-level metrics",
        "| Source | Leads | Warm leads | Applications | Enrollments | Spend | CPL | CAC | Lead→App % | App→Enroll % | Lead→Enroll % | Enrollment share % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for source in result.get("sources", []):
        cur = source.get("current", {})
        share = source.get("share_of_program", {}).get("enrollments")
        lines.append(
            "| {source} | {leads} | {warm} | {apps} | {enroll} | {spend} | {cpl} | {cac} | {lta} | {ate} | {lte} | {share} |".format(
                source=str(source.get("source", "")).replace("|", "\\|"),
                leads=_fmt(cur.get("leads")),
                warm=_fmt(cur.get("warm_leads")),
                apps=_fmt(cur.get("applications")),
                enroll=_fmt(cur.get("enrollments")),
                spend=_fmt(cur.get("spends")),
                cpl=_fmt(cur.get("cpl")),
                cac=_fmt(cur.get("cac")),
                lta=_fmt(cur.get("lead_to_application_pct"), suffix="%"),
                ate=_fmt(cur.get("application_to_enrollment_pct"), suffix="%"),
                lte=_fmt(cur.get("lead_to_enrollment_pct"), suffix="%"),
                share=_fmt(share, suffix="%"),
            )
        )

    lines.extend(["", "## Data notes"])
    lines.extend(f"- {note}" for note in result.get("notes", []))
    if result.get("missing_metrics"):
        lines.append("- Metrics unavailable in this dataset: " + ", ".join(result["missing_metrics"]))
    if result.get("planned_data_status"):
        lines.append("- Planned data status values: " + ", ".join(result["planned_data_status"]))
    return "\n".join(lines)


def _fmt(value: Any, suffix: str = "") -> str:
    number = _number(value)
    if number is None:
        return "N/A"
    if suffix == "%":
        return f"{number:,.2f}%"
    return f"{number:,.2f}{suffix}"
