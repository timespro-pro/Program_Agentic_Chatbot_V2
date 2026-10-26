"""
Shared business-analysis definitions and reporting rules.

This file contains:
- metric definitions
- business interpretation rules
- report structure
- analytical priorities
- data availability rules

Numerical calculations are performed in analysis_engine.py.
Business-level interpretation is performed in overall_analysis.py.
"""

# =========================================================
# METRIC DEFINITIONS
# =========================================================

METRIC_DEFINITIONS = {

    # -----------------------------------------------------
    # VOLUME
    # -----------------------------------------------------

    "leads": {
        "label": "Leads",
        "direction": "higher_is_better",
        "category": "volume",
    },

    "warm_leads": {
        "label": "Warm Leads",
        "direction": "higher_is_better",
        "category": "volume",
    },

    "applications": {
        "label": "Applications",
        "direction": "higher_is_better",
        "category": "volume",
    },

    "enrollments": {
        "label": "Enrollments",
        "direction": "higher_is_better",
        "category": "volume",
    },

    # -----------------------------------------------------
    # REVENUE / GSV
    # -----------------------------------------------------

    "gsv_cr": {
        "label": "GSV",
        "direction": "higher_is_better",
        "category": "revenue",
    },

    # -----------------------------------------------------
    # COST
    # -----------------------------------------------------

    "spends": {
        "label": "Spend",
        "direction": "context_dependent",
        "category": "cost",
    },

    # -----------------------------------------------------
    # EFFICIENCY
    # -----------------------------------------------------

    "cpl": {
        "label": "CPL",
        "direction": "lower_is_better",
        "category": "efficiency",
    },

    "cac": {
        "label": "CAC",
        "direction": "lower_is_better",
        "category": "efficiency",
    },

    # -----------------------------------------------------
    # FUNNEL
    # -----------------------------------------------------

    "lead_to_application_pct": {
        "label": "Lead → Application",
        "direction": "higher_is_better",
        "category": "conversion",
    },

    "application_to_enrollment_pct": {
        "label": "Application → Enrollment",
        "direction": "higher_is_better",
        "category": "conversion",
    },

    "lead_to_enrollment_pct": {
        "label": "Lead → Enrollment",
        "direction": "higher_is_better",
        "category": "conversion",
    },

    # -----------------------------------------------------
    # PAID PERFORMANCE
    # -----------------------------------------------------

    "paid_leads": {
        "label": "Paid Leads",
        "direction": "higher_is_better",
        "category": "paid_volume",
    },

    "paid_enrollments": {
        "label": "Paid Enrollments",
        "direction": "higher_is_better",
        "category": "paid_volume",
    },

    "paid_gsv_cr": {
        "label": "Paid GSV",
        "direction": "higher_is_better",
        "category": "paid_revenue",
    },

    "paid_spends": {
        "label": "Paid Spend",
        "direction": "context_dependent",
        "category": "paid_cost",
    },

    "paid_cpl": {
        "label": "Paid CPL",
        "direction": "lower_is_better",
        "category": "paid_efficiency",
    },

    "paid_cac": {
        "label": "Paid CAC",
        "direction": "lower_is_better",
        "category": "paid_efficiency",
    },

    # -----------------------------------------------------
    # PAID MIX
    # -----------------------------------------------------

    "paid_lead_share_pct": {
        "label": "Paid Lead Share",
        "direction": "context_dependent",
        "category": "mix",
    },

    "paid_enrollment_share_pct": {
        "label": "Paid Enrollment Share",
        "direction": "context_dependent",
        "category": "mix",
    },

    "paid_gsv_share_pct": {
        "label": "Paid GSV Share",
        "direction": "context_dependent",
        "category": "mix",
    },

    "paid_spend_share_pct": {
        "label": "Paid Spend Share",
        "direction": "context_dependent",
        "category": "mix",
    },

    # -----------------------------------------------------
    # PLAN
    # -----------------------------------------------------

    "Planned_Budget": {
        "label": "Planned Budget",
        "direction": "context_dependent",
        "category": "plan",
    },

    "Planned_Lead": {
        "label": "Planned Leads",
        "direction": "higher_is_better",
        "category": "plan",
    },

    "Planned_Application": {
        "label": "Planned Applications",
        "direction": "higher_is_better",
        "category": "plan",
    },

    "Planned_Enrollment": {
        "label": "Planned Enrollments",
        "direction": "higher_is_better",
        "category": "plan",
    },

    "Planned_GSV": {
        "label": "Planned GSV",
        "direction": "higher_is_better",
        "category": "plan",
    },

}


# =========================================================
# ANALYSIS PRIORITIES
# =========================================================

ANALYSIS_PRIORITIES = [

    "overall_performance",

    "business_decline_diagnosis",

    "plan_vs_actual",

    "previous_batch",

    "program_diagnosis",

    "cac_cpl_drivers",

    "funnel",

    "contribution",

    "category",

    "paid_performance",

    "paid_mix",

    "ranking",

    "outliers",

    "management",

]


# =========================================================
# REPORT STRUCTURE
# =========================================================

REPORT_SECTIONS = [

    "1. Executive Summary",

    "2. Current Batch vs Previous Batch",

    "3. Plan vs Actual",

    "4. Where Performance Fell Off",

    "5. Program-Level Comparison",

    "6. Funnel Diagnosis",

    "7. CAC / CPL Diagnosis",

    "8. GSV and Spend Diagnosis",

    "9. Paid Performance and Mix",

    "10. Category-Level Performance",

    "11. Facts, Observations and Hypotheses",

    "12. Finance Investigation",

    "13. Marketing Investigation",

    "14. Management Takeaway",

]


# =========================================================
# BUSINESS PROBLEM
# =========================================================

BUSINESS_PROBLEM_STATEMENT = """
Identify where business performance is improving or declining,
where exactly the deterioration is happening, what the supplied
data shows as the strongest performance signals, what possible
drivers should be investigated, and what Finance, Marketing and
Management should investigate or act on first.
"""


# =========================================================
# DATA HIERARCHY
# =========================================================

DATA_HIERARCHY = [

    "Portfolio",

    "Category",

    "Program",

    "Batch",

    "Funnel",

]


# =========================================================
# AVAILABLE DATA DIMENSIONS
# =========================================================

AVAILABLE_DIMENSIONS = [

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


# =========================================================
# CURRENT DATA AVAILABLE
# =========================================================

CURRENT_DATA_METRICS = [

    "leads",

    "warm_leads",

    "applications",

    "enrollments",

    "gsv_cr",

    "spends",

    "paid_leads",

    "paid_enrollments",

    "paid_gsv_cr",

    "paid_spends",

]


# =========================================================
# PREVIOUS BATCH DATA AVAILABLE
# =========================================================

PREVIOUS_BATCH_METRICS = [

    "previous_batch_leads",

    "previous_batch_warm_leads",

    "previous_batch_applications",

    "previous_batch_enrollments",

    "previous_batch_gsv_cr",

    "previous_batch_spends",

    "previous_batch_paid_leads",

    "previous_batch_paid_enrollments",

    "previous_batch_paid_gsv_cr",

    "previous_batch_paid_spends",

]


# =========================================================
# FUTURE / SEPARATE ANALYSIS
# =========================================================

SEPARATE_ANALYSIS = {

    "source_analysis": {
        "available_now": False,
        "module": "source_analysis.py",
        "description": (
            "Source/channel-level performance analysis."
        ),
    },

    "batch_history": {
        "available_now": True,
        "module": "overall_analysis.py",
        "description": (
            "Current batch versus supplied previous-batch data."
        ),
    },

    "price_analysis": {
        "available_now": False,
        "module": "price_analysis.py",
        "description": (
            "Price-point / fee-change impact analysis."
        ),
    },

    "advisor_analysis": {
        "available_now": False,
        "module": "advisor_analysis.py",
        "description": (
            "Advisor mix analysis."
        ),
    },

    "competition_analysis": {
        "available_now": False,
        "module": "category_analysis.py",
        "description": (
            "Internal and competition program analysis."
        ),
    },

}


# =========================================================
# SYSTEM REPORT RULES
# =========================================================

SYSTEM_REPORT_RULES = """

IMPORTANT ANALYTICAL RULES

Python has already calculated the numerical metrics.

Do NOT recalculate numbers.

Use the supplied values exactly.

---------------------------------------------------------
CAC
---------------------------------------------------------

CAC =
total recorded spend / total enrollments.

Never average row-level CAC values.

Never average source-level CAC values.

---------------------------------------------------------
CPL
---------------------------------------------------------

CPL =
total recorded spend / total leads.

Never average row-level CPL values.

Never average source-level CPL values.

---------------------------------------------------------
MISSING DATA
---------------------------------------------------------

Missing spend means unavailable.

It does NOT mean zero.

Missing GSV means unavailable.

It does NOT mean zero GSV.

Missing paid metrics mean unavailable paid data.

Do not infer zero unless the data explicitly contains zero.

---------------------------------------------------------
FUNNEL
---------------------------------------------------------

Conversion movement must be expressed in
percentage points when comparing conversion rates.

For example:

10% → 8%

means:

-2 percentage points

and not:

-2%.

---------------------------------------------------------
PLAN VS PREVIOUS BATCH
---------------------------------------------------------

Better than plan does not automatically mean
better than the previous batch.

Always distinguish:

Actual vs Plan

from:

Current Batch vs Previous Batch.

---------------------------------------------------------
SPEND
---------------------------------------------------------

Higher spend is not automatically bad.

Spend must be evaluated against:

- leads
- applications
- enrollments
- GSV
- CPL
- CAC
- conversion

---------------------------------------------------------
GSV
---------------------------------------------------------

GSV is a revenue/output metric.

Evaluate GSV together with:

- enrollments
- program fees
- spend
- CAC
- planned GSV

Do not claim that GSV movement was caused by
pricing unless pricing data supports that conclusion.

---------------------------------------------------------
PAID PERFORMANCE
---------------------------------------------------------

Paid performance should be evaluated separately from
overall performance.

Compare:

- paid leads
- paid enrollments
- paid GSV
- paid spend
- paid CPL
- paid CAC
- paid mix

Do not assume that a higher paid share is automatically
better or worse.

---------------------------------------------------------
FACTS
---------------------------------------------------------

Facts are directly supported by supplied data.

Example:

"CAC increased by 12% versus the previous batch."

---------------------------------------------------------
OBSERVATIONS
---------------------------------------------------------

Observations describe relationships visible in the data.

Example:

"Enrollments declined while spend increased."

---------------------------------------------------------
HYPOTHESES
---------------------------------------------------------

Hypotheses are possible explanations that require
investigation.

Do not present hypotheses as facts.

---------------------------------------------------------
CAUSALITY
---------------------------------------------------------

Do not claim that the following caused a change unless
the supplied data proves it:

- lead quality
- campaign quality
- seasonality
- competition
- pricing
- sales operations
- advisor performance
- market conditions
- product positioning
- marketing quality
- source quality

These may be investigation hypotheses only.

---------------------------------------------------------
PREVIOUS BATCH
---------------------------------------------------------

Use the supplied previous_batch_* fields.

Do not invent a previous batch.

Do not reconstruct a previous batch from assumptions
when the required data is unavailable.

---------------------------------------------------------
SOURCE ANALYSIS
---------------------------------------------------------

Source-wise analysis is a separate analysis module.

Do not invent source-level conclusions in the overall
analysis when source-level data has not been supplied.

---------------------------------------------------------
CATEGORY
---------------------------------------------------------

Category-level analysis can be performed when category
is available.

Do not infer competition or internal program information
from category alone.

---------------------------------------------------------
MANAGEMENT
---------------------------------------------------------

Prioritize issues by:

1. Business impact
2. Enrollment impact
3. GSV impact
4. CAC / CPL deterioration
5. Plan gap
6. Previous-batch deterioration
7. Concentration of the problem

Recommendations must identify what Finance,
Marketing or Management should investigate next.
"""
