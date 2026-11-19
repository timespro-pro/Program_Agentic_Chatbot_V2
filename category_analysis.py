"""Category-level performance analysis for Program Performance Analysis Agent V2."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


ADDITIVE_METRICS = [
    "leads", "warm_leads", "applications", "enrollments", "gsv_cr", "spends",
    "Planned_Budget", "Planned_Lead", "Planned_Application", "Planned_Enrollment",
    "Planned_GSV", "paid_leads", "paid_enrollments", "paid_gsv_cr", "paid_spends",
    "planned_paid_leads", "planned_paid_enrollments", "planned_paid_gsv",
    "planned_paid_spends",
]
PROGRAM_ATTRIBUTES = [
    "tech_management", "category", "financial_year", "program_status_f",
    "institute_name_new", "New_Repeat", "business_line", "program_type",
    "program_zone", "program_tier", "program_name",
]
DISPLAY_NAMES = {
    "leads": "Leads", "warm_leads": "Warm Leads", "applications": "Applications",
    "enrollments": "Enrollments", "gsv_cr": "GSV", "spends": "Spend",
    "Planned_Budget": "Planned Budget", "Planned_Lead": "Planned Leads",
    "Planned_Application": "Planned Applications", "Planned_Enrollment": "Planned Enrollments",
    "Planned_GSV": "Planned GSV", "paid_leads": "Paid Leads",
    "paid_enrollments": "Paid Enrollments", "paid_gsv_cr": "Paid GSV",
    "paid_spends": "Paid Spend", "planned_paid_leads": "Planned Paid Leads",
    "planned_paid_enrollments": "Planned Paid Enrollments",
    "planned_paid_gsv": "Planned Paid GSV", "planned_paid_spends": "Planned Paid Spend",
}


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string").str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False).str.replace("%", "", regex=False),
        errors="coerce",
    )


def prepare_category_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Collapse source rows into one program row; additive metrics are summed by source."""
    if raw is None or raw.empty:
        return pd.DataFrame(), {"raw_rows": 0, "programs": 0, "missing_columns": []}
    frame = raw.copy()
    required = {"program_code", "tech_management"}
    missing = sorted(required - set(frame.columns))
    if missing:
        return pd.DataFrame(), {"raw_rows": len(frame), "programs": 0, "missing_columns": missing}

    frame["program_code"] = frame["program_code"].astype("string").str.strip()
    frame = frame[frame["program_code"].notna() & frame["program_code"].ne("")]
    metric_cols = [c for c in ADDITIVE_METRICS if c in frame.columns]
    for col in metric_cols:
        frame[col] = _clean_numeric(frame[col])

    # Each program's dimension values were checked in the supplied CSV; retain first
    # non-null value after collapsing to one record per program.
    agg = {c: "first" for c in PROGRAM_ATTRIBUTES if c in frame.columns}
    agg.update({c: (lambda s: s.sum(min_count=1)) for c in metric_cols})
    programs = frame.groupby("program_code", as_index=False, dropna=False).agg(agg)

    # Keep readable labels and stable sort order.
    for col in ["tech_management", "category", "financial_year", "program_status_f",
                "New_Repeat", "business_line", "program_type", "program_zone", "program_tier"]:
        if col in programs.columns:
            programs[col] = programs[col].astype("string").fillna("Unknown").str.strip()
            programs.loc[programs[col].eq(""), col] = "Unknown"

    programs["program_count"] = 1
    for numerator, denominator, label in [
        ("warm_leads", "leads", "L2W"),
        ("applications", "leads", "L2A"),
        ("applications", "warm_leads", "W2A"),
        ("enrollments", "applications", "A2E"),
        ("enrollments", "leads", "L2E"),
    ]:
        if numerator in programs.columns and denominator in programs.columns:
            programs[label] = np.where(programs[denominator] > 0,
                                       programs[numerator] / programs[denominator], np.nan)

    coverage = {}
    for c in metric_cols:
        coverage[c] = float(frame[c].notna().mean()) if len(frame) else np.nan
    meta = {
        "raw_rows": int(len(frame)),
        "programs": int(programs["program_code"].nunique()),
        "source_rows": int(frame["source_created"].nunique()) if "source_created" in frame.columns else None,
        "metric_coverage": coverage,
        "missing_columns": [],
    }
    return programs, meta


def _sum_if_available(frame: pd.DataFrame, col: str) -> float:
    if col not in frame.columns:
        return np.nan
    return frame[col].sum(min_count=1)


def _safe_rate(num, den):
    if pd.notna(num) and pd.notna(den) and den > 0:
        return num / den
    return np.nan


def render_category_analysis(st, raw_df: pd.DataFrame) -> None:
    st.markdown("## Category Analysis")
    st.caption(
        "Compare performance across the primary tech_management groups and track each group "
        "by financial year. Program-level metrics are consolidated from source rows before category aggregation."
    )
    programs, meta = prepare_category_data(raw_df)
    if programs.empty:
        if meta.get("missing_columns"):
            st.error("Category data is missing required columns: " + ", ".join(meta["missing_columns"]))
        else:
            st.warning("No category data is available.")
        return

    # Filters
    def options(col):
        if col not in programs.columns:
            return []
        return sorted(programs[col].dropna().astype(str).unique().tolist())

    with st.expander("Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            selected_tm = st.multiselect("Tech management", options("tech_management"),
                                         default=options("tech_management"), key="cat_tm_filter")
        with f2:
            selected_fy = st.multiselect("Financial year", options("financial_year"),
                                         default=options("financial_year"), key="cat_fy_filter")
        with f3:
            selected_category = st.multiselect("Category (secondary)", options("category"),
                                               default=options("category"), key="cat_category_filter")
        f4, f5, f6 = st.columns(3)
        with f4:
            selected_status = st.multiselect("Program status", options("program_status_f"),
                                             default=options("program_status_f"), key="cat_status_filter")
        with f5:
            selected_nr = st.multiselect("New / Repeat", options("New_Repeat"),
                                         default=options("New_Repeat"), key="cat_nr_filter")
        with f6:
            selected_business = st.multiselect("Business line", options("business_line"),
                                                default=options("business_line"), key="cat_business_filter")

    filtered = programs.copy()
    for col, selected in [
        ("tech_management", selected_tm), ("financial_year", selected_fy),
        ("category", selected_category), ("program_status_f", selected_status),
        ("New_Repeat", selected_nr), ("business_line", selected_business),
    ]:
        if col in filtered.columns and selected:
            filtered = filtered[filtered[col].astype(str).isin(selected)]
        elif col in filtered.columns and not selected:
            filtered = filtered.iloc[0:0]

    if filtered.empty:
        st.info("No programs match the selected filters.")
        return

    st.markdown("### Performance overview")
    kpi_cols = st.columns(5)
    kpis = [
        ("Programs", filtered["program_code"].nunique(), "{:,.0f}"),
        ("Leads", _sum_if_available(filtered, "leads"), "{:,.0f}"),
        ("Applications", _sum_if_available(filtered, "applications"), "{:,.0f}"),
        ("Enrollments", _sum_if_available(filtered, "enrollments"), "{:,.0f}"),
        ("GSV", _sum_if_available(filtered, "gsv_cr"), "₹{:,.2f} Cr"),
    ]
    for col, (label, value, fmt) in zip(kpi_cols, kpis):
        col.metric(label, "N/A" if pd.isna(value) else fmt.format(value))

    spend = _sum_if_available(filtered, "spends")
    spend_coverage = filtered["spends"].notna().mean() if "spends" in filtered else np.nan
    st.caption(
        f"Spend: {'N/A' if pd.isna(spend) else f'₹{spend:,.2f}'} · "
        f"Spend availability across program records: {'N/A' if pd.isna(spend_coverage) else f'{spend_coverage:.1%}'}"
    )

    st.markdown("### 1. Category comparison")
    metrics_to_show = [c for c in [
        "leads", "warm_leads", "applications", "enrollments", "gsv_cr", "spends",
        "L2W", "L2A", "W2A", "A2E", "L2E"
    ] if c in filtered.columns]
    grouped = filtered.groupby("tech_management", dropna=False).agg(
        Programs=("program_code", "nunique"),
        **{c: (c, lambda s: s.sum(min_count=1)) for c in metrics_to_show if c not in {"L2W","L2A","W2A","A2E","L2E"}}
    ).reset_index()
    # Ratios are recomputed from aggregated counts, not averaged from program rates.
    for label, n, d in [("L2W","warm_leads","leads"),("L2A","applications","leads"),
                        ("W2A","applications","warm_leads"),("A2E","enrollments","applications"),
                        ("L2E","enrollments","leads")]:
        if n in grouped and d in grouped:
            grouped[label] = np.where(grouped[d] > 0, grouped[n] / grouped[d], np.nan)
    for label, numerator, denominator in [
        ("Lead share","leads","leads"), ("Enrollment share","enrollments","enrollments"),
        ("GSV share","gsv_cr","gsv_cr")
    ]:
        if numerator in grouped:
            total = grouped[numerator].sum(min_count=1)
            grouped[label] = grouped[numerator] / total if pd.notna(total) and total > 0 else np.nan

    show = grouped.copy()
    for c in ["L2W","L2A","W2A","A2E","L2E","Lead share","Enrollment share","GSV share"]:
        if c in show:
            show[c] = show[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
    st.dataframe(show, use_container_width=True, hide_index=True)
    if "enrollments" in grouped:
        st.bar_chart(grouped.set_index("tech_management")[["enrollments"]])

    st.markdown("### 2. Financial-year comparison")
    fy_metric = st.selectbox(
        "Metric for financial-year trend",
        [c for c in ["leads","applications","enrollments","gsv_cr","spends","L2A","L2E"] if c in filtered.columns],
        format_func=lambda c: DISPLAY_NAMES.get(c, c),
        key="cat_fy_metric",
    )
    fy_group = filtered.groupby(["financial_year","tech_management"], dropna=False)
    if fy_metric in {"L2A","L2E"}:
        numerator, denominator = (("applications","leads") if fy_metric == "L2A" else ("enrollments","leads"))
        if numerator in filtered and denominator in filtered:
            fy_table = fy_group[[numerator, denominator]].sum(min_count=1).reset_index()
            fy_table[fy_metric] = np.where(fy_table[denominator] > 0,
                                            fy_table[numerator] / fy_table[denominator], np.nan)
            pivot = fy_table.pivot(index="financial_year", columns="tech_management", values=fy_metric)
            st.line_chart(pivot)
            st.dataframe(pivot.style.format("{:.1%}"), use_container_width=True)
        else:
            st.info("Required numerator/denominator columns are not present.")
    else:
        fy_table = fy_group[fy_metric].sum(min_count=1).reset_index()
        pivot = fy_table.pivot(index="financial_year", columns="tech_management", values=fy_metric)
        st.line_chart(pivot)
        st.dataframe(pivot, use_container_width=True)

    st.markdown("### 3. Plan vs actual")
    plan_pairs = [
        ("leads","Planned_Lead","Leads"), ("applications","Planned_Application","Applications"),
        ("enrollments","Planned_Enrollment","Enrollments"), ("gsv_cr","Planned_GSV","GSV"),
        ("spends","Planned_Budget","Spend / Budget"),
    ]
    plan_rows = []
    for actual_col, plan_col, label in plan_pairs:
        if actual_col in filtered and plan_col in filtered:
            actual_val = _sum_if_available(filtered, actual_col)
            plan_val = _sum_if_available(filtered, plan_col)
            attainment = _safe_rate(actual_val, plan_val)
            plan_rows.append({"Metric": label, "Actual": actual_val, "Plan": plan_val,
                              "Attainment": attainment,
                              "Variance": actual_val - plan_val if pd.notna(actual_val) and pd.notna(plan_val) else np.nan})
    if plan_rows:
        plan_df = pd.DataFrame(plan_rows)
        display_plan = plan_df.copy()
        display_plan["Attainment"] = display_plan["Attainment"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        st.dataframe(display_plan, use_container_width=True, hide_index=True)
        st.caption("Plan and actual values are summed from source rows to program totals. Missing plan values remain missing, not zero.")
    else:
        st.info("Plan/actual fields are unavailable in this dataset.")

    st.markdown("### 4. Funnel performance")
    funnel_rows = []
    for label, n, d in [("Lead → Warm Lead","warm_leads","leads"),
                        ("Lead → Application","applications","leads"),
                        ("Warm Lead → Application","applications","warm_leads"),
                        ("Application → Enrollment","enrollments","applications"),
                        ("Lead → Enrollment","enrollments","leads")]:
        if n in filtered and d in filtered:
            funnel_rows.append({"Funnel stage": label, "Numerator": _sum_if_available(filtered,n),
                                "Denominator": _sum_if_available(filtered,d),
                                "Conversion": _safe_rate(_sum_if_available(filtered,n), _sum_if_available(filtered,d))})
    if funnel_rows:
        funnel = pd.DataFrame(funnel_rows)
        funnel["Conversion"] = funnel["Conversion"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
        st.dataframe(funnel, use_container_width=True, hide_index=True)

    st.markdown("### 5. Category business diagnosis")
    diagnosis_rows = grouped.copy()
    if "L2E" in diagnosis_rows and diagnosis_rows["L2E"].notna().any():
        valid_l2e = diagnosis_rows.dropna(subset=["L2E"])
        strongest = valid_l2e.loc[valid_l2e["L2E"].idxmax()]
        weakest = valid_l2e.loc[valid_l2e["L2E"].idxmin()]
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Strongest lead-to-enrollment conversion**")
            st.write(
                f"{strongest['tech_management']}: {strongest['L2E']:.2%} "
                f"({strongest.get('enrollments', 0):,.0f} enrollments from "
                f"{strongest.get('leads', 0):,.0f} leads)."
            )
        with d2:
            st.markdown("**Weakest lead-to-enrollment conversion**")
            st.write(
                f"{weakest['tech_management']}: {weakest['L2E']:.2%} "
                f"({weakest.get('enrollments', 0):,.0f} enrollments from "
                f"{weakest.get('leads', 0):,.0f} leads)."
            )
        st.caption(
            "These are descriptive comparisons. Conversion differences do not establish "
            "why a category performs differently."
        )
    if "spends" in grouped and "enrollments" in grouped:
        spend_coverage_by_cat = filtered.groupby("tech_management")["spends"].apply(
            lambda s: s.notna().mean()
        )
        diagnosis_rows["Spend coverage"] = diagnosis_rows["tech_management"].map(spend_coverage_by_cat)
        diagnosis_rows["CAC"] = np.where(
            diagnosis_rows["enrollments"] > 0,
            diagnosis_rows["spends"] / diagnosis_rows["enrollments"],
            np.nan,
        )
        st.markdown("**Commercial efficiency (interpret alongside spend coverage)**")
        eff = diagnosis_rows[["tech_management", "spends", "enrollments", "CAC", "Spend coverage"]].copy()
        eff["Spend coverage"] = eff["Spend coverage"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        st.dataframe(eff, use_container_width=True, hide_index=True)

    st.markdown("### 6. Program drill-down")
    drill_cols = [c for c in [
        "program_code","program_name","tech_management","category","financial_year",
        "program_status_f","New_Repeat","leads","warm_leads","applications","enrollments",
        "gsv_cr","spends","Planned_Lead","Planned_Application","Planned_Enrollment","Planned_GSV"
    ] if c in filtered.columns]
    st.dataframe(filtered[drill_cols].sort_values(
        "enrollments" if "enrollments" in drill_cols else "program_code",
        ascending=False if "enrollments" in drill_cols else True,
        na_position="last"
    ), use_container_width=True, hide_index=True)

    st.markdown("### Data quality and coverage")
    c1, c2, c3 = st.columns(3)
    c1.metric("Source rows", f"{meta['raw_rows']:,}")
    c2.metric("Unique programs", f"{meta['programs']:,}")
    c3.metric("Source dimensions", f"{meta.get('source_rows') or 0:,}")
    coverage_rows = []
    for col in ADDITIVE_METRICS:
        if col in raw_df.columns:
            coverage_rows.append({
                "Metric": DISPLAY_NAMES.get(col, col),
                "Non-missing source-row coverage": f"{raw_df[col].notna().mean():.1%}",
            })
    if coverage_rows:
        st.dataframe(pd.DataFrame(coverage_rows), use_container_width=True, hide_index=True)
    st.caption(
        "Interpretation note: source-level actuals and plans are summed to program level; "
        "program attributes are retained once. Ratios are calculated from aggregated totals. "
        "Spend and plan comparisons should be interpreted in light of the coverage table."
    )
