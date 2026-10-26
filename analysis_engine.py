"""
Deterministic business-performance analysis engine.

The LLM does NOT calculate business metrics.

Python calculates:
- volume
- GSV
- spend
- CPL
- CAC
- funnel conversion
- current vs previous batch
- plan vs actual
- paid performance
- paid mix
- program performance
- category performance
- deterioration
- improvement
- contribution / concentration

Primary data source:
    Batch_wise_program_data.csv

Source-wise analysis is intentionally kept separate and can be
implemented later in source_analysis.py.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import pandas as pd


# =========================================================
# COLUMN DEFINITIONS
# =========================================================

NUMERIC_COLUMNS = [

    # Current actual
    "total_program_fees",
    "leads",
    "warm_leads",
    "applications",
    "enrollments",
    "gsv_cr",
    "spends",

    # Current plan
    "Planned_Budget",
    "Planned_Lead",
    "Planned_Application",
    "Planned_Enrollment",
    "Planned_GSV",

    # Current paid
    "paid_leads",
    "paid_enrollments",
    "paid_gsv_cr",
    "paid_spends",

    # Current paid plan
    "planned_paid_leads",
    "planned_paid_enrollments",
    "planned_paid_gsv",
    "planned_paid_spends",

    # Previous batch actual
    "previous_batch_leads",
    "previous_batch_warm_leads",
    "previous_batch_applications",
    "previous_batch_enrollments",
    "previous_batch_gsv_cr",
    "previous_batch_spends",

    # Previous batch plan
    "previous_batch_Planned_Budget",
    "previous_batch_Planned_Lead",
    "previous_batch_Planned_Application",
    "previous_batch_Planned_Enrollment",
    "previous_batch_Planned_GSV",

    # Previous batch paid
    "previous_batch_paid_leads",
    "previous_batch_paid_enrollments",
    "previous_batch_paid_gsv_cr",
    "previous_batch_paid_spends",

    # Previous batch paid plan
    "previous_batch_planned_paid_leads",
    "previous_batch_planned_paid_enrollments",
    "previous_batch_planned_paid_gsv",
    "previous_batch_planned_paid_spends",
]


ACTUAL_ADDITIVE = [
    "leads",
    "applications",
    "enrollments",
    "warm_leads",
    "gsv_cr",
    "spends",
]


PLAN_ADDITIVE = [
    "Planned_Budget",
    "Planned_Lead",
    "Planned_Application",
    "Planned_Enrollment",
    "Planned_GSV",
]


PAID_ACTUAL_ADDITIVE = [
    "paid_leads",
    "paid_enrollments",
    "paid_gsv_cr",
    "paid_spends",
]


PAID_PLAN_ADDITIVE = [
    "planned_paid_leads",
    "planned_paid_enrollments",
    "planned_paid_gsv",
    "planned_paid_spends",
]


PREVIOUS_ACTUAL_COLUMNS = {
    "leads": "previous_batch_leads",
    "warm_leads": "previous_batch_warm_leads",
    "applications": "previous_batch_applications",
    "enrollments": "previous_batch_enrollments",
    "gsv_cr": "previous_batch_gsv_cr",
    "spends": "previous_batch_spends",
}


PREVIOUS_PAID_COLUMNS = {
    "paid_leads": "previous_batch_paid_leads",
    "paid_enrollments": "previous_batch_paid_enrollments",
    "paid_gsv_cr": "previous_batch_paid_gsv_cr",
    "paid_spends": "previous_batch_paid_spends",
}


# =========================================================
# COMPLETE DATA REQUIREMENTS
# =========================================================

COMPLETE_ROW_FIELDS = [
    "program_code",
    "previous_batch_code",
    "sales_start_date",
    "category",
    "financial_year",
    "program_status_f",
    "total_program_fees",
    "source_created",
    "New_Repeat",
    "leads",
    "warm_leads",
    "applications",
    "enrollments",
    "gsv_cr",
    "spends",
    "Planned_Budget",
    "Planned_Lead",
    "Planned_Application",
    "Planned_Enrollment",
    "Planned_GSV",
    "Planned_Data_Status",
]


# =========================================================
# DATA PREPARATION
# =========================================================

def prepare_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if df is None:
        return pd.DataFrame()

    x = df.copy()

    # -----------------------------------------------------
    # Clean column names
    # -----------------------------------------------------

    x.columns = [
        str(column).strip()
        for column in x.columns
    ]

    # -----------------------------------------------------
    # Numeric conversion
    # -----------------------------------------------------

    for column in NUMERIC_COLUMNS:

        if column not in x.columns:
            continue

        x[column] = pd.to_numeric(
            x[column]
            .astype(str)
            .str.replace(
                ",",
                "",
                regex=False,
            )
            .str.replace(
                "₹",
                "",
                regex=False,
            )
            .str.strip(),
            errors="coerce",
        )

    # -----------------------------------------------------
    # Date conversion
    # -----------------------------------------------------

    if "sales_start_date" in x.columns:

        x["sales_start_date"] = pd.to_datetime(
            x["sales_start_date"],
            errors="coerce",
        )

    # -----------------------------------------------------
    # Text cleanup
    # -----------------------------------------------------

    text_columns = [
        "program_code",
        "tech_management",
        "previous_batch_code",
        "category",
        "financial_year",
        "program_status_f",
        "source_created",
        "New_Repeat",
        "Planned_Data_Status",
    ]

    for column in text_columns:

        if column in x.columns:

            x[column] = x[column].apply(
                lambda value:
                str(value).strip()
                if pd.notna(value)
                else None
            )

    return x


# =========================================================
# COMPLETE ROW FILTER
# =========================================================

def apply_source_filter(
    df: pd.DataFrame,
    complete_rows_only: bool = True,
):
    """
    Backward-compatible filtering function.

    For the new overall dataset, the filter is deliberately
    less aggressive than the old source-row filter.

    Missing spend is NOT interpreted as zero.
    """

    x = prepare_data(df)

    original_rows = len(x)

    if not complete_rows_only:

        return (
            x,
            {
                "enabled": False,
                "original_rows": original_rows,
                "retained_rows": original_rows,
                "removed_rows": 0,
                "required_fields": COMPLETE_ROW_FIELDS,
            },
        )

    available_fields = [
        column
        for column in COMPLETE_ROW_FIELDS
        if column in x.columns
    ]

    if not available_fields:

        return (
            x,
            {
                "enabled": True,
                "original_rows": original_rows,
                "retained_rows": original_rows,
                "removed_rows": 0,
                "required_fields": [],
            },
        )

    # -----------------------------------------------------
    # IMPORTANT
    #
    # Do not require every metric to be populated.
    #
    # A program can legitimately have:
    # - no paid data
    # - no GSV yet
    # - no spend yet
    #
    # These are unavailable values, not bad rows.
    # -----------------------------------------------------

    identity_fields = [
        column
        for column in [
            "program_code",
            "category",
            "previous_batch_code",
        ]
        if column in x.columns
    ]

    if identity_fields:

        mask = (
            x[identity_fields]
            .notna()
            .all(axis=1)
        )

        filtered = x.loc[
            mask
        ].copy()

    else:

        filtered = x.copy()

    return (
        filtered,
        {
            "enabled": True,
            "original_rows": original_rows,
            "retained_rows": len(filtered),
            "removed_rows": (
                original_rows - len(filtered)
            ),
            "required_fields": identity_fields,
        },
    )


# =========================================================
# NUMERIC HELPERS
# =========================================================

def _num(
    value: Any,
):

    if value is None:
        return None

    try:

        if pd.isna(value):
            return None

        value = float(value)

        if not math.isfinite(value):
            return None

        if value.is_integer():
            return int(value)

        return round(
            value,
            4,
        )

    except (
        TypeError,
        ValueError,
    ):

        return None


def _sum_col(
    rows: pd.DataFrame,
    column: str,
):

    if (
        rows.empty
        or column not in rows.columns
    ):

        return None

    value = pd.to_numeric(
        rows[column],
        errors="coerce",
    ).sum(
        min_count=1
    )

    return _num(value)


def _first(
    rows: pd.DataFrame,
    column: str,
):

    if column not in rows.columns:
        return None

    values = rows[column].dropna()

    if values.empty:
        return None

    value = values.iloc[0]

    if isinstance(
        value,
        pd.Timestamp,
    ):

        return value.strftime(
            "%Y-%m-%d"
        )

    return _num(value) if not isinstance(
        value,
        str,
    ) else value


def _change(
    current,
    previous,
):

    if (
        current is None
        or previous is None
    ):

        return None

    return round(
        current - previous,
        2,
    )


def _pct_change(
    current,
    previous,
):

    if (
        current is None
        or previous is None
        or previous == 0
    ):

        return None

    return round(
        (
            (
                current - previous
            )
            / abs(previous)
        )
        * 100,
        2,
    )


def _ratio(
    numerator,
    denominator,
):

    if (
        numerator is None
        or denominator is None
        or denominator == 0
    ):

        return None

    return round(
        numerator / denominator,
        4,
    )


# =========================================================
# STATUS
# =========================================================

def _status(
    metric_key,
    current,
    previous,
):

    if (
        current is None
        or previous is None
    ):

        return "not_available"

    if current == previous:
        return "flat"

    # Higher is better
    higher_is_better = {
        "leads",
        "applications",
        "enrollments",
        "warm_leads",
        "gsv_cr",
        "lead_to_application_pct",
        "application_to_enrollment_pct",
        "lead_to_enrollment_pct",
    }

    # Lower is better
    lower_is_better = {
        "cpl",
        "cac",
    }

    if metric_key in higher_is_better:

        if current > previous:
            return "improved"

        return "deteriorated"

    if metric_key in lower_is_better:

        if current < previous:
            return "improved"

        return "deteriorated"

    return (
        "increased"
        if current > previous
        else "decreased"
    )


# =========================================================
# SCOPE
# =========================================================

def _scope(
    df: pd.DataFrame,
    program_code,
):

    if not program_code:
        return df

    if "program_code" not in df.columns:
        return df.iloc[0:0]

    code = (
        str(program_code)
        .strip()
        .upper()
    )

    return df[
        df["program_code"]
        .astype(str)
        .str.strip()
        .str.upper()
        == code
    ]


# =========================================================
# ACTUAL METRICS
# =========================================================

def calculate_actuals(
    rows: pd.DataFrame,
):

    result = {
        column:
        _sum_col(
            rows,
            column,
        )
        for column in ACTUAL_ADDITIVE
    }

    spend = result.get(
        "spends"
    )

    leads = result.get(
        "leads"
    )

    applications = result.get(
        "applications"
    )

    enrollments = result.get(
        "enrollments"
    )

    # -----------------------------------------------------
    # CPL
    # -----------------------------------------------------

    result["cpl"] = _ratio(
        spend,
        leads,
    )

    # -----------------------------------------------------
    # CAC
    # -----------------------------------------------------

    result["cac"] = _ratio(
        spend,
        enrollments,
    )

    # -----------------------------------------------------
    # Funnel
    # -----------------------------------------------------

    result[
        "lead_to_application_pct"
    ] = (
        round(
            applications / leads * 100,
            2,
        )
        if applications is not None
        and leads not in (
            None,
            0,
        )
        else None
    )

    result[
        "application_to_enrollment_pct"
    ] = (
        round(
            enrollments
            / applications
            * 100,
            2,
        )
        if enrollments is not None
        and applications not in (
            None,
            0,
        )
        else None
    )

    result[
        "lead_to_enrollment_pct"
    ] = (
        round(
            enrollments / leads * 100,
            2,
        )
        if enrollments is not None
        and leads not in (
            None,
            0,
        )
        else None
    )

    return result


# Backward-compatible alias
_actual = calculate_actuals


# =========================================================
# PAID ACTUAL METRICS
# =========================================================

def calculate_paid_actuals(
    rows: pd.DataFrame,
):

    result = {
        key:
        _sum_col(
            rows,
            key,
        )
        for key in PAID_ACTUAL_ADDITIVE
    }

    result["paid_cpl"] = _ratio(
        result.get("paid_spends"),
        result.get("paid_leads"),
    )

    result["paid_cac"] = _ratio(
        result.get("paid_spends"),
        result.get("paid_enrollments"),
    )

    return result


# =========================================================
# PAID MIX
# =========================================================

def calculate_paid_mix(
    actual: dict,
):

    return {
        "paid_lead_share_pct": (
            round(
                actual["paid_leads"]
                / actual["leads"]
                * 100,
                2,
            )
            if actual.get("paid_leads")
            is not None
            and actual.get("leads")
            not in (None, 0)
            else None
        ),

        "paid_enrollment_share_pct": (
            round(
                actual["paid_enrollments"]
                / actual["enrollments"]
                * 100,
                2,
            )
            if actual.get(
                "paid_enrollments"
            )
            is not None
            and actual.get(
                "enrollments"
            )
            not in (None, 0)
            else None
        ),

        "paid_gsv_share_pct": (
            round(
                actual["paid_gsv_cr"]
                / actual["gsv_cr"]
                * 100,
                2,
            )
            if actual.get(
                "paid_gsv_cr"
            )
            is not None
            and actual.get(
                "gsv_cr"
            )
            not in (None, 0)
            else None
        ),

        "paid_spend_share_pct": (
            round(
                actual["paid_spends"]
                / actual["spends"]
                * 100,
                2,
            )
            if actual.get(
                "paid_spends"
            )
            is not None
            and actual.get(
                "spends"
            )
            not in (None, 0)
            else None
        ),
    }


# =========================================================
# PLAN
# =========================================================

def calculate_plan(
    rows: pd.DataFrame,
):

    if rows.empty:

        return {
            column: None
            for column in PLAN_ADDITIVE
        }

    result = {}

    for column in PLAN_ADDITIVE:

        result[column] = _sum_col(
            rows,
            column,
        )

    return result


_plan = calculate_plan


# =========================================================
# PAID PLAN
# =========================================================

def calculate_paid_plan(
    rows: pd.DataFrame,
):

    return {
        column:
        _sum_col(
            rows,
            column,
        )
        for column in PAID_PLAN_ADDITIVE
    }


# =========================================================
# PLAN VS ACTUAL
# =========================================================

def compare_to_plan(
    actual: dict,
    plan: dict,
):

    planned_budget = plan.get(
        "Planned_Budget"
    )

    planned_leads = plan.get(
        "Planned_Lead"
    )

    planned_applications = plan.get(
        "Planned_Application"
    )

    planned_enrollments = plan.get(
        "Planned_Enrollment"
    )

    planned_gsv = plan.get(
        "Planned_GSV"
    )

    planned_cpl = _ratio(
        planned_budget,
        planned_leads,
    )

    planned_cac = _ratio(
        planned_budget,
        planned_enrollments,
    )

    def metric(
        actual_value,
        planned_value,
    ):

        return {
            "actual": actual_value,
            "planned": planned_value,
            "variance": _change(
                actual_value,
                planned_value,
            ),
            "variance_pct": _pct_change(
                actual_value,
                planned_value,
            ),
        }

    return {

        "budget": metric(
            actual.get("spends"),
            planned_budget,
        ),

        "leads": metric(
            actual.get("leads"),
            planned_leads,
        ),

        "applications": metric(
            actual.get("applications"),
            planned_applications,
        ),

        "enrollments": metric(
            actual.get("enrollments"),
            planned_enrollments,
        ),

        "gsv_cr": metric(
            actual.get("gsv_cr"),
            planned_gsv,
        ),

        "cpl": metric(
            actual.get("cpl"),
            planned_cpl,
        ),

        "cac": metric(
            actual.get("cac"),
            planned_cac,
        ),
    }


# =========================================================
# CURRENT BATCH LIST
# =========================================================

def _current_batch_codes(
    df: pd.DataFrame,
):

    if (
        df.empty
        or "program_code" not in df.columns
    ):

        return []

    all_codes = set(
        df["program_code"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    previous_codes = set()

    if "previous_batch_code" in df.columns:

        previous_codes = set(
            df["previous_batch_code"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )

    current = sorted(
        all_codes - previous_codes
    )

    if not current:
        current = sorted(
            all_codes
        )

    return current


# =========================================================
# OLD BATCH LIST
# =========================================================

def _old_batch_codes(
    df: pd.DataFrame,
):

    if (
        df.empty
        or "previous_batch_code"
        not in df.columns
    ):

        return []

    return sorted(
        set(
            df["previous_batch_code"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )
    )


# =========================================================
# PREVIOUS BATCH FOR PROGRAM
# =========================================================

def get_previous_batch_for_program(
    df: pd.DataFrame,
    program_code: str,
):

    rows = _scope(
        df,
        program_code,
    )

    if rows.empty:
        return None

    if (
        "previous_batch_code"
        not in rows.columns
    ):

        return None

    values = (
        rows["previous_batch_code"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    return (
        values[0]
        if values
        else None
    )


# =========================================================
# PREVIOUS BATCH DATA FROM NEW CSV
# =========================================================

def calculate_previous_actuals(
    rows: pd.DataFrame,
):

    result = {}

    for metric, column in (
        PREVIOUS_ACTUAL_COLUMNS.items()
    ):

        result[metric] = _sum_col(
            rows,
            column,
        )

    result["cpl"] = _ratio(
        result.get("spends"),
        result.get("leads"),
    )

    result["cac"] = _ratio(
        result.get("spends"),
        result.get("enrollments"),
    )

    leads = result.get(
        "leads"
    )

    applications = result.get(
        "applications"
    )

    enrollments = result.get(
        "enrollments"
    )

    result[
        "lead_to_application_pct"
    ] = (
        round(
            applications / leads * 100,
            2,
        )
        if applications is not None
        and leads not in (
            None,
            0,
        )
        else None
    )

    result[
        "application_to_enrollment_pct"
    ] = (
        round(
            enrollments
            / applications
            * 100,
            2,
        )
        if enrollments is not None
        and applications not in (
            None,
            0,
        )
        else None
    )

    result[
        "lead_to_enrollment_pct"
    ] = (
        round(
            enrollments / leads * 100,
            2,
        )
        if enrollments is not None
        and leads not in (
            None,
            0,
        )
        else None
    )

    return result


# =========================================================
# BATCH COMPARISON
# =========================================================

def compare_batches(
    current: dict,
    previous: Optional[dict],
):

    if not previous:

        return []

    metrics = [

        (
            "Spend",
            "spends",
            "cost",
        ),

        (
            "Leads",
            "leads",
            "volume",
        ),

        (
            "Applications",
            "applications",
            "volume",
        ),

        (
            "Enrollments",
            "enrollments",
            "volume",
        ),

        (
            "Warm Leads",
            "warm_leads",
            "volume",
        ),

        (
            "GSV",
            "gsv_cr",
            "revenue",
        ),

        (
            "CPL",
            "cpl",
            "efficiency",
        ),

        (
            "CAC",
            "cac",
            "efficiency",
        ),

        (
            "Lead → Application",
            "lead_to_application_pct",
            "conversion",
        ),

        (
            "Application → Enrollment",
            "application_to_enrollment_pct",
            "conversion",
        ),

        (
            "Lead → Enrollment",
            "lead_to_enrollment_pct",
            "conversion",
        ),
    ]

    result = []

    for (
        label,
        key,
        kind,
    ) in metrics:

        current_value = current.get(
            key
        )

        previous_value = previous.get(
            key
        )

        result.append({

            "metric": label,

            "key": key,

            "kind": kind,

            "current": current_value,

            "previous": previous_value,

            "change": _change(
                current_value,
                previous_value,
            ),

            "change_pct": _pct_change(
                current_value,
                previous_value,
            ),

            "status": _status(
                key,
                current_value,
                previous_value,
            ),
        })

    return result


_compare = compare_batches


# =========================================================
# PROGRAM METADATA
# =========================================================

def get_program_metadata(
    rows: pd.DataFrame,
):

    if rows.empty:
        return {}

    metadata = {}

    columns = [
        "program_code",
        "tech_management",
        "previous_batch_code",
        "sales_start_date",
        "category",
        "financial_year",
        "program_status_f",
        "total_program_fees",
        "source_created",
        "New_Repeat",
        "Planned_Data_Status",
    ]

    for column in columns:

        if column not in rows.columns:
            continue

        values = (
            rows[column]
            .dropna()
            .unique()
        )

        if len(values) == 0:
            metadata[column] = None

        elif len(values) == 1:

            value = values[0]

            if isinstance(
                value,
                pd.Timestamp,
            ):

                value = value.strftime(
                    "%Y-%m-%d"
                )

            metadata[column] = value

        else:

            metadata[column] = [
                (
                    value.strftime(
                        "%Y-%m-%d"
                    )
                    if isinstance(
                        value,
                        pd.Timestamp,
                    )
                    else value
                )
                for value in values
            ]

    return metadata


# =========================================================
# PROGRAM ANALYSIS
# =========================================================

def get_program_analysis(
    df: pd.DataFrame,
    program_code,
    old_batch_code=None,
):

    df = prepare_data(df)

    current_rows = _scope(
        df,
        program_code,
    )

    if current_rows.empty:

        return {
            "found": False,
            "program_code": program_code,
        }

    # -----------------------------------------------------
    # Current
    # -----------------------------------------------------

    current = calculate_actuals(
        current_rows
    )

    current_paid = calculate_paid_actuals(
        current_rows
    )

    current_plan = calculate_plan(
        current_rows
    )

    current_paid_plan = calculate_paid_plan(
        current_rows
    )

    # -----------------------------------------------------
    # Previous batch
    #
    # IMPORTANT:
    #
    # The new CSV already contains previous batch
    # metrics. Therefore we use those fields rather
    # than requiring the old CSV.
    # -----------------------------------------------------

    previous = calculate_previous_actuals(
        current_rows
    )

    previous_available = any(
        value is not None
        for value in previous.values()
    )

    previous_code = (
        old_batch_code
        or get_previous_batch_for_program(
            df,
            program_code,
        )
    )

    # -----------------------------------------------------
    # Comparisons
    # -----------------------------------------------------

    comparisons = (
        compare_batches(
            current,
            previous,
        )
        if previous_available
        else []
    )

    # -----------------------------------------------------
    # Plan comparison
    # -----------------------------------------------------

    plan_comparison = compare_to_plan(
        current,
        current_plan,
    )

    # -----------------------------------------------------
    # Paid
    # -----------------------------------------------------

    paid_mix = calculate_paid_mix(
        current_paid
    )

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    metadata = get_program_metadata(
        current_rows
    )

    return {

        "found": True,

        "program_code": program_code,

        "previous_batch_code":
            previous_code,

        "metadata":
            metadata,

        "current_actual":
            {
                **current,
                **current_paid,
                **paid_mix,
            },

        "previous_actual":
            previous
            if previous_available
            else None,

        "batch_comparison":
            comparisons,

        "plan":
            current_plan,

        "plan_comparison":
            plan_comparison,

        "paid_plan":
            current_paid_plan,

        "previous_batch_available":
            previous_available,

        # Source analysis deliberately separated.
        "source_comparison": [],
    }


# =========================================================
# PORTFOLIO ANALYSIS
# =========================================================

def get_portfolio_analysis(
    df: pd.DataFrame,
):

    df = prepare_data(df)

    if df.empty:

        return {
            "found": False,
            "message": "No data available.",
        }

    current = calculate_actuals(
        df
    )

    current_paid = calculate_paid_actuals(
        df
    )

    current_plan = calculate_plan(
        df
    )

    current_paid_plan = calculate_paid_plan(
        df
    )

    previous = calculate_previous_actuals(
        df
    )

    previous_available = any(
        value is not None
        for value in previous.values()
    )

    comparisons = (
        compare_batches(
            current,
            previous,
        )
        if previous_available
        else []
    )

    plan_comparison = compare_to_plan(
        current,
        current_plan,
    )

    paid_mix = calculate_paid_mix(
        current_paid
    )

    program_codes = sorted(
        df["program_code"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    ) if "program_code" in df.columns else []

    return {

        "found": True,

        "scope": "portfolio",

        "program_count":
            len(program_codes),

        "program_codes":
            program_codes,

        "current_actual":
            {
                **current,
                **current_paid,
                **paid_mix,
            },

        "previous_actual":
            previous
            if previous_available
            else None,

        "batch_comparison":
            comparisons,

        "plan":
            current_plan,

        "plan_comparison":
            plan_comparison,

        "paid_plan":
            current_paid_plan,

        "previous_batch_available":
            previous_available,
    }


# =========================================================
# BATCH SUMMARY
# =========================================================

def get_batch_summary(
    df: pd.DataFrame,
    program_code=None,
):

    if program_code:

        return get_program_analysis(
            df,
            program_code,
        )

    return get_portfolio_analysis(
        df
    )


# =========================================================
# VISUAL REPORT
# =========================================================

def get_visual_report_data(
    df: pd.DataFrame,
    program_code,
    old_batch_code=None,
):

    if not program_code:

        return {
            "found": False,
            "message":
                "Select a New Batch.",
        }

    analysis = get_program_analysis(
        df,
        program_code,
        old_batch_code=old_batch_code,
    )

    if not analysis.get(
        "found"
    ):

        return analysis

    comparisons = analysis.get(
        "batch_comparison",
        [],
    )

    # -----------------------------------------------------
    # Business-impact priority
    # -----------------------------------------------------

    priority = {

        "cac": 10,

        "cpl": 9,

        "lead_to_enrollment_pct": 9,

        "application_to_enrollment_pct": 8,

        "lead_to_application_pct": 8,

        "enrollments": 7,

        "applications": 6,

        "leads": 5,

        "gsv_cr": 5,

        "spends": 2,

        "warm_leads": 3,
    }

    deterioration = [
        row
        for row in comparisons
        if row.get("status")
        == "deteriorated"
    ]

    improvement = [
        row
        for row in comparisons
        if row.get("status")
        == "improved"
    ]

    deterioration.sort(
        key=lambda row: (
            priority.get(
                row.get("key"),
                0,
            ),
            abs(
                row.get(
                    "change_pct"
                )
                or 0
            ),
        ),
        reverse=True,
    )

    improvement.sort(
        key=lambda row:
        abs(
            row.get(
                "change_pct"
            )
            or 0
        ),
        reverse=True,
    )

    return {

        "found": True,

        "program_code":
            program_code,

        "previous_batch_code":
            analysis.get(
                "previous_batch_code"
            ),

        "current":
            analysis.get(
                "current_actual"
            ),

        "previous":
            analysis.get(
                "previous_actual"
            ),

        "comparisons":
            comparisons,

        "key_dropoffs":
            deterioration[:5],

        "improvements":
            improvement[:5],

        "funnel": [
            row
            for row in comparisons
            if row.get("key")
            in (
                "leads",
                "applications",
                "enrollments",
                "lead_to_application_pct",
                "application_to_enrollment_pct",
                "lead_to_enrollment_pct",
            )
        ],

        "source_comparison":
            [],

        "plan":
            analysis.get(
                "plan",
                {},
            ),

        "plan_comparison":
            analysis.get(
                "plan_comparison",
                {},
            ),

        "metadata":
            analysis.get(
                "metadata",
                {},
            ),
    }


# =========================================================
# PROGRAM PORTFOLIO SUMMARY
# =========================================================

def portfolio_program_summary(
    df: pd.DataFrame,
):

    df = prepare_data(df)

    if (
        df.empty
        or "program_code"
        not in df.columns
    ):

        return pd.DataFrame()

    rows = []

    for (
        program_code,
        program_df,
    ) in df.groupby(
        "program_code",
        dropna=False,
    ):

        actual = calculate_actuals(
            program_df
        )

        plan = calculate_plan(
            program_df
        )

        previous = calculate_previous_actuals(
            program_df
        )

        metadata = get_program_metadata(
            program_df
        )

        rows.append({

            "program_code":
                program_code,

            "category":
                metadata.get(
                    "category"
                ),

            "financial_year":
                metadata.get(
                    "financial_year"
                ),

            "program_status_f":
                metadata.get(
                    "program_status_f"
                ),

            "New_Repeat":
                metadata.get(
                    "New_Repeat"
                ),

            "total_program_fees":
                metadata.get(
                    "total_program_fees"
                ),

            "leads":
                actual.get(
                    "leads"
                ),

            "applications":
                actual.get(
                    "applications"
                ),

            "enrollments":
                actual.get(
                    "enrollments"
                ),

            "gsv_cr":
                actual.get(
                    "gsv_cr"
                ),

            "spends":
                actual.get(
                    "spends"
                ),

            "cpl":
                actual.get(
                    "cpl"
                ),

            "cac":
                actual.get(
                    "cac"
                ),

            "planned_leads":
                plan.get(
                    "Planned_Lead"
                ),

            "planned_applications":
                plan.get(
                    "Planned_Application"
                ),

            "planned_enrollments":
                plan.get(
                    "Planned_Enrollment"
                ),

            "planned_gsv_cr":
                plan.get(
                    "Planned_GSV"
                ),

            "planned_budget":
                plan.get(
                    "Planned_Budget"
                ),

            "lead_variance_pct":
                _pct_change(
                    actual.get(
                        "leads"
                    ),
                    plan.get(
                        "Planned_Lead"
                    ),
                ),

            "application_variance_pct":
                _pct_change(
                    actual.get(
                        "applications"
                    ),
                    plan.get(
                        "Planned_Application"
                    ),
                ),

            "enrollment_variance_pct":
                _pct_change(
                    actual.get(
                        "enrollments"
                    ),
                    plan.get(
                        "Planned_Enrollment"
                    ),
                ),

            "gsv_variance_pct":
                _pct_change(
                    actual.get(
                        "gsv_cr"
                    ),
                    plan.get(
                        "Planned_GSV"
                    ),
                ),

            "budget_variance_pct":
                _pct_change(
                    actual.get(
                        "spends"
                    ),
                    plan.get(
                        "Planned_Budget"
                    ),
                ),

            "previous_enrollments":
                previous.get(
                    "enrollments"
                ),

            "previous_cac":
                previous.get(
                    "cac"
                ),

            "enrollment_change_pct":
                _pct_change(
                    actual.get(
                        "enrollments"
                    ),
                    previous.get(
                        "enrollments"
                    ),
                ),

            "cac_change_pct":
                _pct_change(
                    actual.get(
                        "cac"
                    ),
                    previous.get(
                        "cac"
                    ),
                ),
        })

    return pd.DataFrame(
        rows
    )


# =========================================================
# PERFORMANCE DIPS
# =========================================================

def identify_performance_dips(
    df: pd.DataFrame,
    top_n: int = 10,
):

    summary = portfolio_program_summary(
        df
    )

    if summary.empty:

        return {
            "below_plan_enrollment": [],
            "below_plan_gsv": [],
            "high_cac": [],
            "high_cpl": [],
            "cac_increase": [],
            "enrollment_decline": [],
        }

    result = {}

    # -----------------------------------------------------
    # Below-plan enrollment
    # -----------------------------------------------------

    enrollment = summary[
        summary[
            "enrollment_variance_pct"
        ].notna()
    ].sort_values(
        "enrollment_variance_pct",
        ascending=True,
    )

    result[
        "below_plan_enrollment"
    ] = enrollment.head(
        top_n
    ).to_dict(
        "records"
    )

    # -----------------------------------------------------
    # Below-plan GSV
    # -----------------------------------------------------

    gsv = summary[
        summary[
            "gsv_variance_pct"
        ].notna()
    ].sort_values(
        "gsv_variance_pct",
        ascending=True,
    )

    result[
        "below_plan_gsv"
    ] = gsv.head(
        top_n
    ).to_dict(
        "records"
    )

    # -----------------------------------------------------
    # Highest CAC
    # -----------------------------------------------------

    high_cac = summary[
        summary["cac"].notna()
    ].sort_values(
        "cac",
        ascending=False,
    )

    result[
        "high_cac"
    ] = high_cac.head(
        top_n
    ).to_dict(
        "records"
    )

    # -----------------------------------------------------
    # Highest CPL
    # -----------------------------------------------------

    high_cpl = summary[
        summary["cpl"].notna()
    ].sort_values(
        "cpl",
        ascending=False,
    )

    result[
        "high_cpl"
    ] = high_cpl.head(
        top_n
    ).to_dict(
        "records"
    )

    # -----------------------------------------------------
    # CAC increase vs previous
    # -----------------------------------------------------

    cac_increase = summary[
        summary[
            "cac_change_pct"
        ].notna()
        & (
            summary[
                "cac_change_pct"
            ] > 0
        )
    ].sort_values(
        "cac_change_pct",
        ascending=False,
    )

    result[
        "cac_increase"
    ] = cac_increase.head(
        top_n
    ).to_dict(
        "records"
    )

    # -----------------------------------------------------
    # Enrollment decline vs previous
    # -----------------------------------------------------

    enrollment_decline = summary[
        summary[
            "enrollment_change_pct"
        ].notna()
        & (
            summary[
                "enrollment_change_pct"
            ] < 0
        )
    ].sort_values(
        "enrollment_change_pct",
        ascending=True,
    )

    result[
        "enrollment_decline"
    ] = enrollment_decline.head(
        top_n
    ).to_dict(
        "records"
    )

    return result


# =========================================================
# CATEGORY SUMMARY
# =========================================================

def category_analysis(
    df: pd.DataFrame,
):

    df = prepare_data(df)

    if (
        df.empty
        or "category"
        not in df.columns
    ):

        return pd.DataFrame()

    rows = []

    for (
        category,
        category_df,
    ) in df.groupby(
        "category",
        dropna=False,
    ):

        actual = calculate_actuals(
            category_df
        )

        plan = calculate_plan(
            category_df
        )

        rows.append({

            "category":
                category,

            "program_count":
                (
                    category_df[
                        "program_code"
                    ]
                    .nunique()
                    if "program_code"
                    in category_df.columns
                    else len(
                        category_df
                    )
                ),

            "leads":
                actual.get(
                    "leads"
                ),

            "applications":
                actual.get(
                    "applications"
                ),

            "enrollments":
                actual.get(
                    "enrollments"
                ),

            "gsv_cr":
                actual.get(
                    "gsv_cr"
                ),

            "spends":
                actual.get(
                    "spends"
                ),

            "cpl":
                actual.get(
                    "cpl"
                ),

            "cac":
                actual.get(
                    "cac"
                ),

            "planned_leads":
                plan.get(
                    "Planned_Lead"
                ),

            "planned_enrollments":
                plan.get(
                    "Planned_Enrollment"
                ),

            "planned_gsv_cr":
                plan.get(
                    "Planned_GSV"
                ),

            "planned_budget":
                plan.get(
                    "Planned_Budget"
                ),

            "enrollment_variance_pct":
                _pct_change(
                    actual.get(
                        "enrollments"
                    ),
                    plan.get(
                        "Planned_Enrollment"
                    ),
                ),

            "gsv_variance_pct":
                _pct_change(
                    actual.get(
                        "gsv_cr"
                    ),
                    plan.get(
                        "Planned_GSV"
                    ),
                ),
        })

    return pd.DataFrame(
        rows
    )


# Backward-compatible alias
cluster_analysis = category_analysis


# =========================================================
# CONTRIBUTION ANALYSIS
# =========================================================

def contribution_analysis(
    df: pd.DataFrame,
    metric: str = "enrollments",
    top_n: int = 10,
):

    summary = portfolio_program_summary(
        df
    )

    if summary.empty:
        return []

    if metric not in summary.columns:
        return []

    values = pd.to_numeric(
        summary[metric],
        errors="coerce",
    )

    total = values.sum()

    if total == 0:
        return []

    result = summary.copy()

    result[
        "portfolio_share_pct"
    ] = (
        result[metric]
        / total
        * 100
    )

    result = result.sort_values(
        metric,
        ascending=False,
    )

    return result.head(
        top_n
    ).to_dict(
        "records"
    )


# =========================================================
# DATA PROFILE
# =========================================================

def get_data_profile(
    df: pd.DataFrame,
):

    df = prepare_data(df)

    return {

        "rows":
            len(df),

        "columns":
            len(df.columns),

        "program_count":
            (
                df[
                    "program_code"
                ]
                .nunique()
                if "program_code"
                in df.columns
                else 0
            ),

        "category_count":
            (
                df[
                    "category"
                ]
                .nunique()
                if "category"
                in df.columns
                else 0
            ),

        "financial_year_count":
            (
                df[
                    "financial_year"
                ]
                .nunique()
                if "financial_year"
                in df.columns
                else 0
            ),

        "date_min":
            (
                df[
                    "sales_start_date"
                ].min()
                if "sales_start_date"
                in df.columns
                else None
            ),

        "date_max":
            (
                df[
                    "sales_start_date"
                ].max()
                if "sales_start_date"
                in df.columns
                else None
            ),
    }


# =========================================================
# OVERALL ANALYSIS CONTEXT
# =========================================================

def build_analysis_context(
    df: pd.DataFrame,
    program_code=None,
    old_batch_code=None,
    check_df=None,
    question="",
):

    # check_df is intentionally retained in the signature
    # for backward compatibility with the old app.
    #
    # It is NOT used.
    #
    # The new architecture has one overall data source.

    if program_code:

        analysis = get_program_analysis(
            df,
            program_code,
            old_batch_code=old_batch_code,
        )

        return {

            "scope":
                "program",

            "selected_new_batch":
                program_code,

            "selected_old_batch":
                old_batch_code,

            "program":
                analysis,

            "rules": {

                "CPL":
                    "total spend / total leads",

                "CAC":
                    "total spend / total enrollments",

                "GSV":
                    "sum of gsv_cr",

                "missing_spend":
                    "unavailable, not zero",

                "conversion_change":
                    "percentage-point movement",
            },
        }

    # -----------------------------------------------------
    # Portfolio
    # -----------------------------------------------------

    portfolio = get_portfolio_analysis(
        df
    )

    current_codes = _current_batch_codes(
        df
    )

    programs = []

    for code in current_codes:

        analysis = get_program_analysis(
            df,
            code,
        )

        programs.append({

            "program_code":
                code,

            "previous_batch_code":
                analysis.get(
                    "previous_batch_code"
                ),

            "metadata":
                analysis.get(
                    "metadata",
                    {},
                ),

            "current_actual":
                analysis.get(
                    "current_actual"
                ),

            "previous_actual":
                analysis.get(
                    "previous_actual"
                ),

            "batch_comparison":
                analysis.get(
                    "batch_comparison",
                    [],
                ),

            "plan_comparison":
                analysis.get(
                    "plan_comparison",
                    {},
                ),
        })

    return {

        "scope":
            "portfolio",

        "portfolio":
            portfolio,

        "program_codes":
            current_codes,

        "programs":
            programs,
    }


# =========================================================
# BACKWARD COMPATIBILITY
# =========================================================

def compare_previous_batch(
    df: pd.DataFrame,
    program_code=None,
):

    if program_code:

        analysis = get_program_analysis(
            df,
            program_code,
        )

        return analysis.get(
            "batch_comparison",
            [],
        )

    portfolio = get_portfolio_analysis(
        df
    )

    return portfolio.get(
        "batch_comparison",
        [],
    )


def funnel_analysis(
    df: pd.DataFrame,
    program_code=None,
):

    if program_code:

        analysis = get_program_analysis(
            df,
            program_code,
        )

        actual = analysis.get(
            "current_actual",
            {},
        )

    else:

        portfolio = get_portfolio_analysis(
            df
        )

        actual = portfolio.get(
            "current_actual",
            {},
        )

    return {

        "leads":
            actual.get(
                "leads"
            ),

        "applications":
            actual.get(
                "applications"
            ),

        "enrollments":
            actual.get(
                "enrollments"
            ),

        "lead_to_application_pct":
            actual.get(
                "lead_to_application_pct"
            ),

        "application_to_enrollment_pct":
            actual.get(
                "application_to_enrollment_pct"
            ),

        "lead_to_enrollment_pct":
            actual.get(
                "lead_to_enrollment_pct"
            ),
    }


def source_analysis(
    df: pd.DataFrame,
    program_code=None,
):

    # -----------------------------------------------------
    # Source-wise analysis intentionally moved out of the
    # overall engine.
    #
    # Implement later in:
    #
    #     source_analysis.py
    #
    # -----------------------------------------------------

    return {
        "available": False,
        "message": (
            "Source-wise analysis is handled separately "
            "and is not part of the current overall analysis."
        ),
    }


def check_tool_analysis(
    *args,
    **kwargs,
):

    return {
        "available": False,
        "message": (
            "Program Check Tool is no longer a separate "
            "data source. The overall analysis now uses "
            "Batch_wise_program_data.csv."
        ),
    }
