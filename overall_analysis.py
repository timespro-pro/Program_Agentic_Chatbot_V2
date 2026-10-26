"""
Overall business-performance analysis.

This module sits between:

analysis_engine.py
        ↓
overall_analysis.py
        ↓
agent.py

analysis_engine.py performs numerical calculations.

This module interprets those calculations into
business-performance signals.

Source-wise analysis is intentionally excluded and
will later live in source_analysis.py.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from analysis_engine import (
    get_portfolio_analysis,
    get_program_analysis,
    portfolio_program_summary,
    identify_performance_dips,
    category_analysis,
    contribution_analysis,
)

from analysis_points import (
    ANALYSIS_PRIORITIES,
    BUSINESS_PROBLEM_STATEMENT,
    REPORT_SECTIONS,
)


# =========================================================
# HELPERS
# =========================================================

def _pct(
    value,
):
    """
    Convert decimal ratio to percentage.
    """

    if value is None:
        return None

    try:
        return round(
            float(value) * 100,
            2,
        )

    except (
        TypeError,
        ValueError,
    ):

        return None


def _round(
    value,
    digits=2,
):

    if value is None:
        return None

    try:

        return round(
            float(value),
            digits,
        )

    except (
        TypeError,
        ValueError,
    ):

        return value


def _is_negative(
    value,
):

    return (
        value is not None
        and value < 0
    )


def _is_positive(
    value,
):

    return (
        value is not None
        and value > 0
    )


# =========================================================
# PLAN SIGNALS
# =========================================================

def _build_plan_signals(
    plan_comparison: dict,
):

    signals = []

    # -----------------------------------------------------
    # Enrollments
    # -----------------------------------------------------

    enrollment = plan_comparison.get(
        "enrollments",
        {},
    )

    variance = enrollment.get(
        "variance_pct"
    )

    if _is_negative(
        variance
    ):

        signals.append({

            "area":
                "enrollments",

            "severity":
                "high",

            "comparison":
                "actual_vs_plan",

            "variance_pct":
                variance,

            "variance_pct_display":
                _pct(variance),

            "signal":
                "below_plan",

            "interpretation":
                (
                    "Actual enrollments are below "
                    "planned enrollments."
                ),
        })

    # -----------------------------------------------------
    # GSV
    # -----------------------------------------------------

    gsv = plan_comparison.get(
        "gsv_cr",
        {},
    )

    variance = gsv.get(
        "variance_pct"
    )

    if _is_negative(
        variance
    ):

        signals.append({

            "area":
                "gsv",

            "severity":
                "high",

            "comparison":
                "actual_vs_plan",

            "variance_pct":
                variance,

            "variance_pct_display":
                _pct(variance),

            "signal":
                "below_plan",

            "interpretation":
                (
                    "Actual GSV is below "
                    "planned GSV."
                ),
        })

    # -----------------------------------------------------
    # Leads
    # -----------------------------------------------------

    leads = plan_comparison.get(
        "leads",
        {},
    )

    variance = leads.get(
        "variance_pct"
    )

    if _is_negative(
        variance
    ):

        signals.append({

            "area":
                "leads",

            "severity":
                "medium",

            "comparison":
                "actual_vs_plan",

            "variance_pct":
                variance,

            "variance_pct_display":
                _pct(variance),

            "signal":
                "below_plan",

            "interpretation":
                (
                    "Lead volume is below plan."
                ),
        })

    # -----------------------------------------------------
    # Applications
    # -----------------------------------------------------

    applications = plan_comparison.get(
        "applications",
        {},
    )

    variance = applications.get(
        "variance_pct"
    )

    if _is_negative(
        variance
    ):

        signals.append({

            "area":
                "applications",

            "severity":
                "medium",

            "comparison":
                "actual_vs_plan",

            "variance_pct":
                variance,

            "variance_pct_display":
                _pct(variance),

            "signal":
                "below_plan",

            "interpretation":
                (
                    "Applications are below plan."
                ),
        })

    # -----------------------------------------------------
    # CAC
    # -----------------------------------------------------

    cac = plan_comparison.get(
        "cac",
        {},
    )

    variance = cac.get(
        "variance_pct"
    )

    if _is_positive(
        variance
    ):

        signals.append({

            "area":
                "cac",

            "severity":
                "high",

            "comparison":
                "actual_vs_plan",

            "variance_pct":
                variance,

            "variance_pct_display":
                _pct(variance),

            "signal":
                "worse_than_plan",

            "interpretation":
                (
                    "Actual CAC is higher than "
                    "planned CAC."
                ),
        })

    # -----------------------------------------------------
    # CPL
    # -----------------------------------------------------

    cpl = plan_comparison.get(
        "cpl",
        {},
    )

    variance = cpl.get(
        "variance_pct"
    )

    if _is_positive(
        variance
    ):

        signals.append({

            "area":
                "cpl",

            "severity":
                "medium",

            "comparison":
                "actual_vs_plan",

            "variance_pct":
                variance,

            "variance_pct_display":
                _pct(variance),

            "signal":
                "worse_than_plan",

            "interpretation":
                (
                    "Actual CPL is higher than "
                    "planned CPL."
                ),
        })

    return signals


# =========================================================
# PREVIOUS BATCH SIGNALS
# =========================================================

def _build_previous_batch_signals(
    batch_comparison,
):

    signals = []

    for row in batch_comparison:

        status = row.get(
            "status"
        )

        if status not in (
            "deteriorated",
            "improved",
        ):

            continue

        key = row.get(
            "key"
        )

        change_pct = row.get(
            "change_pct"
        )

        if status == "deteriorated":

            severity = "medium"

            if key in (
                "cac",
                "enrollments",
                "gsv_cr",
            ):

                severity = "high"

            signals.append({

                "area":
                    key,

                "severity":
                    severity,

                "comparison":
                    "current_vs_previous",

                "change_pct":
                    change_pct,

                "signal":
                    "deteriorated",

                "interpretation":
                    (
                        f"{row.get('metric')} "
                        "deteriorated versus the "
                        "previous batch."
                    ),
            })

    return signals


# =========================================================
# PROGRAM-LEVEL DIAGNOSIS
# =========================================================

def _program_attention_scores(
    programs: pd.DataFrame,
):

    if programs.empty:
        return programs

    x = programs.copy()

    # -----------------------------------------------------
    # Score negative enrollment variance
    # -----------------------------------------------------

    x["attention_score"] = 0.0

    if (
        "enrollment_variance_pct"
        in x.columns
    ):

        x["attention_score"] += (
            x[
                "enrollment_variance_pct"
            ]
            .fillna(0)
            .clip(
                upper=0
            )
            .abs()
            * 3
        )

    # -----------------------------------------------------
    # Score CAC deterioration
    # -----------------------------------------------------

    if (
        "cac_change_pct"
        in x.columns
    ):

        x["attention_score"] += (
            x[
                "cac_change_pct"
            ]
            .fillna(0)
            .clip(
                lower=0
            )
            * 3
        )

    # -----------------------------------------------------
    # Score GSV gap
    # -----------------------------------------------------

    if (
        "gsv_variance_pct"
        in x.columns
    ):

        x["attention_score"] += (
            x[
                "gsv_variance_pct"
            ]
            .fillna(0)
            .clip(
                upper=0
            )
            .abs()
            * 2
        )

    # -----------------------------------------------------
    # Score high CAC
    # -----------------------------------------------------

    if (
        "cac"
        in x.columns
    ):

        median_cac = x[
            "cac"
        ].median()

        if pd.notna(
            median_cac
        ):

            x["attention_score"] += (
                (
                    x["cac"]
                    > median_cac
                )
                .astype(int)
                * 1
            )

    return x.sort_values(
        "attention_score",
        ascending=False,
    )


# =========================================================
# CONCENTRATION ANALYSIS
# =========================================================

def _analyze_concentration(
    programs: pd.DataFrame,
    top_n: int = 5,
):

    if programs.empty:

        return {
            "available": False,
            "reason":
                "No program data available.",
        }

    if "enrollments" not in programs.columns:

        return {
            "available": False,
            "reason":
                "Enrollment data unavailable.",
        }

    total_enrollments = pd.to_numeric(
        programs["enrollments"],
        errors="coerce",
    ).sum()

    if not total_enrollments:

        return {
            "available": False,
            "reason":
                "Total enrollments are unavailable or zero.",
        }

    # Weakest programs by enrollment
    weakest = programs.sort_values(
        "enrollments",
        ascending=True,
    ).head(
        top_n
    )

    weak_enrollments = pd.to_numeric(
        weakest["enrollments"],
        errors="coerce",
    ).sum()

    weak_share = (
        weak_enrollments
        / total_enrollments
    )

    return {

        "available": True,

        "top_n":
            top_n,

        "weakest_programs":
            weakest[
                "program_code"
            ].tolist(),

        "weak_enrollments":
            float(
                weak_enrollments
            ),

        "portfolio_enrollments":
            float(
                total_enrollments
            ),

        "weak_enrollment_share_pct":
            round(
                weak_share * 100,
                2,
            ),

    }


# =========================================================
# OVERALL PERFORMANCE
# =========================================================

def analyze_overall_performance(
    df: pd.DataFrame,
):

    portfolio = get_portfolio_analysis(
        df
    )

    programs = portfolio_program_summary(
        df
    )

    performance_dips = identify_performance_dips(
        df
    )

    categories = category_analysis(
        df
    )

    concentration = _analyze_concentration(
        programs
    )

    return {

        "portfolio":
            portfolio,

        "program_performance": {

            "count":
                len(programs),

            "table":
                programs,

        },

        "performance_dips":
            performance_dips,

        "category_performance": {

            "count":
                len(categories),

            "table":
                categories,

        },

        "concentration":
            concentration,

    }


# =========================================================
# BUSINESS DIAGNOSIS
# =========================================================

def diagnose_business_performance(
    analysis: dict,
):

    portfolio = analysis[
        "portfolio"
    ]

    plan_comparison = portfolio.get(
        "plan_comparison",
        {},
    )

    batch_comparison = portfolio.get(
        "batch_comparison",
        [],
    )

    # -----------------------------------------------------
    # Plan signals
    # -----------------------------------------------------

    plan_signals = _build_plan_signals(
        plan_comparison
    )

    # -----------------------------------------------------
    # Previous batch signals
    # -----------------------------------------------------

    previous_signals = (
        _build_previous_batch_signals(
            batch_comparison
        )
    )

    signals = (
        plan_signals
        + previous_signals
    )

    # -----------------------------------------------------
    # Sort by business severity
    # -----------------------------------------------------

    severity_rank = {
        "high": 3,
        "medium": 2,
        "low": 1,
    }

    signals.sort(
        key=lambda item:
        severity_rank.get(
            item.get(
                "severity"
            ),
            0,
        ),
        reverse=True,
    )

    return {

        "signal_count":
            len(signals),

        "signals":
            signals,

        "plan_signals":
            plan_signals,

        "previous_batch_signals":
            previous_signals,

    }


# =========================================================
# PROGRAM DIAGNOSIS
# =========================================================

def diagnose_programs(
    analysis: dict,
    top_n: int = 10,
):

    programs = analysis[
        "program_performance"
    ][
        "table"
    ]

    if programs.empty:

        return {

            "available":
                False,

            "attention_programs":
                [],

        }

    scored = _program_attention_scores(
        programs
    )

    attention = scored.head(
        top_n
    )

    return {

        "available":
            True,

        "attention_programs":
            attention.to_dict(
                "records"
            ),

        "below_plan_enrollment":
            analysis[
                "performance_dips"
            ].get(
                "below_plan_enrollment",
                [],
            )[:top_n],

        "below_plan_gsv":
            analysis[
                "performance_dips"
            ].get(
                "below_plan_gsv",
                [],
            )[:top_n],

        "highest_cac":
            analysis[
                "performance_dips"
            ].get(
                "high_cac",
                [],
            )[:top_n],

        "highest_cpl":
            analysis[
                "performance_dips"
            ].get(
                "high_cpl",
                [],
            )[:top_n],

        "cac_increase":
            analysis[
                "performance_dips"
            ].get(
                "cac_increase",
                [],
            )[:top_n],

        "enrollment_decline":
            analysis[
                "performance_dips"
            ].get(
                "enrollment_decline",
                [],
            )[:top_n],

    }


# =========================================================
# CATEGORY DIAGNOSIS
# =========================================================

def diagnose_categories(
    analysis: dict,
):

    categories = analysis[
        "category_performance"
    ][
        "table"
    ]

    if categories.empty:

        return {

            "available":
                False,

            "categories":
                [],

        }

    result = categories.copy()

    if (
        "enrollment_variance_pct"
        in result.columns
    ):

        result = result.sort_values(
            "enrollment_variance_pct",
            ascending=True,
        )

    return {

        "available":
            True,

        "categories":
            result.to_dict(
                "records"
            ),

    }


# =========================================================
# CAC / CPL DIAGNOSIS
# =========================================================

def diagnose_efficiency(
    analysis: dict,
):

    portfolio = analysis[
        "portfolio"
    ]

    actual = portfolio.get(
        "current_actual",
        {},
    )

    plan = portfolio.get(
        "plan_comparison",
        {},
    )

    result = {

        "actual_cpl":
            actual.get(
                "cpl"
            ),

        "actual_cac":
            actual.get(
                "cac"
            ),

        "planned_cpl":
            plan.get(
                "cpl",
                {}
            ).get(
                "planned"
            ),

        "planned_cac":
            plan.get(
                "cac",
                {}
            ).get(
                "planned"
            ),

        "cpl_variance_pct":
            plan.get(
                "cpl",
                {}
            ).get(
                "variance_pct"
            ),

        "cac_variance_pct":
            plan.get(
                "cac",
                {}
            ).get(
                "variance_pct"
            ),

    }

    # -----------------------------------------------------
    # Interpretation
    # -----------------------------------------------------

    if (
        result["cac_variance_pct"]
        is not None
        and result[
            "cac_variance_pct"
        ] > 0
    ):

        result[
            "primary_efficiency_signal"
        ] = "cac_above_plan"

    elif (
        result["cpl_variance_pct"]
        is not None
        and result[
            "cpl_variance_pct"
        ] > 0
    ):

        result[
            "primary_efficiency_signal"
        ] = "cpl_above_plan"

    else:

        result[
            "primary_efficiency_signal"
        ] = "no_major_efficiency_gap"

    return result


# =========================================================
# FUNNEL DIAGNOSIS
# =========================================================

def diagnose_funnel(
    analysis: dict,
):

    actual = analysis[
        "portfolio"
    ].get(
        "current_actual",
        {},
    )

    previous = analysis[
        "portfolio"
    ].get(
        "previous_actual",
        {},
    )

    funnel = {

        "lead_to_application": {

            "current":
                actual.get(
                    "lead_to_application_pct"
                ),

            "previous":
                (
                    previous.get(
                        "lead_to_application_pct"
                    )
                    if previous
                    else None
                ),

        },

        "application_to_enrollment": {

            "current":
                actual.get(
                    "application_to_enrollment_pct"
                ),

            "previous":
                (
                    previous.get(
                        "application_to_enrollment_pct"
                    )
                    if previous
                    else None
                ),

        },

        "lead_to_enrollment": {

            "current":
                actual.get(
                    "lead_to_enrollment_pct"
                ),

            "previous":
                (
                    previous.get(
                        "lead_to_enrollment_pct"
                    )
                    if previous
                    else None
                ),

        },

    }

    # Add percentage-point movement
    for key, values in funnel.items():

        current = values.get(
            "current"
        )

        previous_value = values.get(
            "previous"
        )

        if (
            current is not None
            and previous_value is not None
        ):

            values[
                "change_percentage_points"
            ] = round(
                current
                - previous_value,
                2,
            )

        else:

            values[
                "change_percentage_points"
            ] = None

    return funnel


# =========================================================
# GSV / SPEND DIAGNOSIS
# =========================================================

def diagnose_gsv_and_spend(
    analysis: dict,
):

    portfolio = analysis[
        "portfolio"
    ]

    actual = portfolio.get(
        "current_actual",
        {},
    )

    previous = portfolio.get(
        "previous_actual",
    )

    plan = portfolio.get(
        "plan_comparison",
        {},
    )

    result = {

        "current_gsv":
            actual.get(
                "gsv_cr"
            ),

        "current_spend":
            actual.get(
                "spends"
            ),

        "previous_gsv":
            (
                previous.get(
                    "gsv_cr"
                )
                if previous
                else None
            ),

        "previous_spend":
            (
                previous.get(
                    "spends"
                )
                if previous
                else None
            ),

        "planned_gsv":
            plan.get(
                "gsv_cr",
                {}
            ).get(
                "planned"
            ),

        "planned_spend":
            plan.get(
                "budget",
                {}
            ).get(
                "planned"
            ),

    }

    # -----------------------------------------------------
    # Current vs previous
    # -----------------------------------------------------

    if (
        result["current_gsv"]
        is not None
        and result["previous_gsv"]
        is not None
    ):

        previous_gsv = (
            result["previous_gsv"]
        )

        if previous_gsv != 0:

            result[
                "gsv_change_pct"
            ] = round(
                (
                    (
                        result[
                            "current_gsv"
                        ]
                        - previous_gsv
                    )
                    / abs(
                        previous_gsv
                    )
                )
                * 100,
                2,
            )

    # -----------------------------------------------------
    # Spend change
    # -----------------------------------------------------

    if (
        result["current_spend"]
        is not None
        and result["previous_spend"]
        is not None
    ):

        previous_spend = (
            result["previous_spend"]
        )

        if previous_spend != 0:

            result[
                "spend_change_pct"
            ] = round(
                (
                    (
                        result[
                            "current_spend"
                        ]
                        - previous_spend
                    )
                    / abs(
                        previous_spend
                    )
                )
                * 100,
                2,
            )

    return result


# =========================================================
# PAID PERFORMANCE
# =========================================================

def diagnose_paid_performance(
    analysis: dict,
):

    actual = analysis[
        "portfolio"
    ].get(
        "current_actual",
        {},
    )

    return {

        "paid_leads":
            actual.get(
                "paid_leads"
            ),

        "paid_enrollments":
            actual.get(
                "paid_enrollments"
            ),

        "paid_gsv_cr":
            actual.get(
                "paid_gsv_cr"
            ),

        "paid_spends":
            actual.get(
                "paid_spends"
            ),

        "paid_cpl":
            actual.get(
                "paid_cpl"
            ),

        "paid_cac":
            actual.get(
                "paid_cac"
            ),

        "paid_lead_share_pct":
            actual.get(
                "paid_lead_share_pct"
            ),

        "paid_enrollment_share_pct":
            actual.get(
                "paid_enrollment_share_pct"
            ),

        "paid_gsv_share_pct":
            actual.get(
                "paid_gsv_share_pct"
            ),

        "paid_spend_share_pct":
            actual.get(
                "paid_spend_share_pct"
            ),

    }


# =========================================================
# CONTRIBUTION
# =========================================================

def analyze_contributions(
    df: pd.DataFrame,
):

    return {

        "enrollment_contributors":
            contribution_analysis(
                df,
                metric="enrollments",
                top_n=10,
            ),

        "gsv_contributors":
            contribution_analysis(
                df,
                metric="gsv_cr",
                top_n=10,
            ),

        "spend_contributors":
            contribution_analysis(
                df,
                metric="spends",
                top_n=10,
            ),

    }


# =========================================================
# SELECTED PROGRAM
# =========================================================

def analyze_selected_program(
    df: pd.DataFrame,
    program_code: str,
    old_batch_code: Optional[str] = None,
):

    analysis = get_program_analysis(
        df,
        program_code,
        old_batch_code=old_batch_code,
    )

    if not analysis.get(
        "found"
    ):

        return {

            "available":
                False,

            "program_code":
                program_code,

        }

    return {

        "available":
            True,

        "program_code":
            program_code,

        "metadata":
            analysis.get(
                "metadata",
                {},
            ),

        "current":
            analysis.get(
                "current_actual",
                {},
            ),

        "previous":
            analysis.get(
                "previous_actual"
            ),

        "previous_batch_code":
            analysis.get(
                "previous_batch_code"
            ),

        "batch_comparison":
            analysis.get(
                "batch_comparison",
                [],
            ),

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

        "paid_plan":
            analysis.get(
                "paid_plan",
                {},
            ),

    }


# =========================================================
# COMPLETE OVERALL ANALYSIS
# =========================================================

def build_overall_analysis(
    df: pd.DataFrame,
    program_code: Optional[str] = None,
    old_batch_code: Optional[str] = None,
):

    # -----------------------------------------------------
    # Base deterministic analysis
    # -----------------------------------------------------

    overall = analyze_overall_performance(
        df
    )

    # -----------------------------------------------------
    # Business diagnosis
    # -----------------------------------------------------

    diagnosis = diagnose_business_performance(
        overall
    )

    # -----------------------------------------------------
    # Program diagnosis
    # -----------------------------------------------------

    programs = diagnose_programs(
        overall
    )

    # -----------------------------------------------------
    # Category diagnosis
    # -----------------------------------------------------

    categories = diagnose_categories(
        overall
    )

    # -----------------------------------------------------
    # Efficiency
    # -----------------------------------------------------

    efficiency = diagnose_efficiency(
        overall
    )

    # -----------------------------------------------------
    # Funnel
    # -----------------------------------------------------

    funnel = diagnose_funnel(
        overall
    )

    # -----------------------------------------------------
    # GSV / Spend
    # -----------------------------------------------------

    gsv_spend = diagnose_gsv_and_spend(
        overall
    )

    # -----------------------------------------------------
    # Paid
    # -----------------------------------------------------

    paid = diagnose_paid_performance(
        overall
    )

    # -----------------------------------------------------
    # Contribution
    # -----------------------------------------------------

    contributions = analyze_contributions(
        df
    )

    result = {

        "business_problem":
            BUSINESS_PROBLEM_STATEMENT,

        "analysis_priorities":
            ANALYSIS_PRIORITIES,

        "report_sections":
            REPORT_SECTIONS,

        "overall":
            overall,

        "business_diagnosis":
            diagnosis,

        "program_diagnosis":
            programs,

        "category_diagnosis":
            categories,

        "efficiency_diagnosis":
            efficiency,

        "funnel_diagnosis":
            funnel,

        "gsv_spend_diagnosis":
            gsv_spend,

        "paid_diagnosis":
            paid,

        "contribution":
            contributions,

        "source_analysis":
            {
                "available": False,

                "message": (
                    "Source-wise analysis is intentionally "
                    "kept separate and will be implemented "
                    "in source_analysis.py."
                ),
            },

    }

    # -----------------------------------------------------
    # Selected program
    # -----------------------------------------------------

    if program_code:

        result[
            "selected_program"
        ] = analyze_selected_program(
            df,
            program_code,
            old_batch_code=old_batch_code,
        )

    return result


# =========================================================
# AGENT CONTEXT
# =========================================================

def build_overall_analysis_context(
    df: pd.DataFrame,
    program_code: Optional[str] = None,
    old_batch_code: Optional[str] = None,
):

    return build_overall_analysis(
        df=df,
        program_code=program_code,
        old_batch_code=old_batch_code,
    )
