"""
Optimal Launch Metrics engine for Program Performance Analysis Agent.

This module is intentionally separate from app.py. It:
1. Infers the next batch code from the selected batch.
2. Builds a comparable historical program-family dataset.
3. Uses a small, guarded ML layer when enough complete observations exist.
4. Falls back to robust historical benchmarks when training data is insufficient.
5. Renders the Optimal Launch Metrics section without requiring changes to the app's theme.

Note: predictions are decision-support estimates, not guaranteed outcomes. The ML layer
is intentionally conservative because aggregated historical batch data is often limited.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


METRIC_CANDIDATES = {
    "Program fee": ["total_program_fees", "program_fee", "fee", "program_fees"],
    "Total spend": ["spends", "Spends_final", "total_spend", "spend"],
    "GCV": ["gsv_cr", "gcv", "GCV", "gross_course_value"],
    "Leads": ["leads", "total_leads"],
    "Warm leads": ["warm_leads", "warm_lead"],
    "Applications": ["applications", "application"],
    "Enrollments": ["enrollments", "enrollment"],
    "Paid spend": ["paid_spends", "paid_spend", "paid_marketing_spend"],
    "Paid leads": ["paid_leads"],
    "Paid enrollments": ["paid_enrollments"],
    "Paid GCV": ["paid_gsv_cr", "paid_gcv"],
}

MONEY_METRICS = {
    "Program fee", "Total spend", "GCV", "Paid spend", "Paid GCV",
    "Estimated paid CPL", "Estimated paid CAC",
}


def infer_next_batch_code(selected_code: str) -> str:
    """Increment the final numeric suffix, preserving zero-padding."""
    code = str(selected_code or "").strip()
    match = re.match(r"^(.*?)(\d+)$", code)
    if not match:
        return ""
    prefix, digits = match.groups()
    next_number = int(digits) + 1
    return f"{prefix}{next_number:0{len(digits)}d}"


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_to_actual = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_actual:
            return lower_to_actual[candidate.lower()]
    return None


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.replace("₹", "", regex=False),
        errors="coerce",
    )


def _program_level_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate repeated source rows to one record per program/batch."""
    if raw_df is None or raw_df.empty or "program_code" not in raw_df.columns:
        return pd.DataFrame()
    frame = raw_df.copy()
    frame["program_code"] = frame["program_code"].astype(str).str.strip()
    frame = frame[frame["program_code"].ne("") & frame["program_code"].ne("nan")]
    # Source-level columns repeat program totals. One row per program prevents source-row
    # duplication from falsely inflating the effective sample size.
    frame = frame.drop_duplicates(subset=["program_code"], keep="first").copy()

    for label, candidates in METRIC_CANDIDATES.items():
        col = _find_column(frame, candidates)
        if col:
            frame[f"__metric__{label}"] = _numeric(frame[col])
    return frame


def _family_prefix(code: str) -> tuple[str, str] | None:
    match = re.match(r"^(.*?)(\d+)$", str(code).strip())
    if not match:
        return None
    return match.group(1), match.group(2)


def _select_history(frame: pd.DataFrame, selected_code: str, category: str = "All categories") -> pd.DataFrame:
    if frame.empty:
        return frame
    family = _family_prefix(selected_code)
    history = frame[frame["program_code"].ne(selected_code)].copy()

    # Prefer same-family batches (e.g. IIMC_SMP_28, IIMC_SMP_27 for IIMC_SMP_29).
    if family:
        prefix, _ = family
        same_family = history[history["program_code"].str.match(rf"^{re.escape(prefix)}\d+$", na=False)]
        if not same_family.empty:
            history = same_family

    category_col = _find_column(history, ["category", "program_category"])
    if category != "All categories" and category_col:
        category_rows = history[history[category_col].astype(str).eq(category)]
        if not category_rows.empty:
            history = category_rows
    return history


def _metric_series(history: pd.DataFrame, label: str) -> pd.Series:
    col = f"__metric__{label}"
    if col not in history.columns:
        return pd.Series(dtype=float)
    return history[col].replace([np.inf, -np.inf], np.nan).dropna()


def _safe_ratio(numerator: float, denominator: float) -> float:
    if pd.notna(numerator) and pd.notna(denominator) and denominator > 0:
        return float(numerator / denominator)
    return float("nan")


def _try_ml_predictions(history: pd.DataFrame, target_labels: list[str]) -> dict[str, float]:
    """
    Conservative ML layer. Uses only numeric features available in historical rows.
    It trains only when enough independent program-level observations exist.
    """
    predictions: dict[str, float] = {}
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
    except Exception:
        return predictions

    # With very few batches, a fitted model is more misleading than useful.
    if len(history) < 12:
        return predictions

    feature_labels = ["Program fee", "Total spend", "Paid spend", "Leads", "Applications", "Paid leads"]
    feature_cols = [f"__metric__{label}" for label in feature_labels if f"__metric__{label}" in history.columns]
    if len(feature_cols) < 2:
        return predictions

    for target in target_labels:
        target_col = f"__metric__{target}"
        if target_col not in history.columns:
            continue
        training = history[feature_cols + [target_col]].replace([np.inf, -np.inf], np.nan)
        training = training.dropna(subset=[target_col])
        if len(training) < 12:
            continue

        X = training[feature_cols]
        y = training[target_col]
        # Do not leak the target into its own feature set.
        safe_features = [c for c in feature_cols if c != target_col]
        if not safe_features:
            continue
        X = training[safe_features]
        try:
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestRegressor(
                    n_estimators=160,
                    min_samples_leaf=2,
                    random_state=42,
                    max_features=1.0,
                ),
            )
            model.fit(X, y)
            # A future-batch input does not yet exist. For an initial forecast, predict at
            # the median observed feature profile and clearly label as model-assisted.
            baseline = X.median(numeric_only=True).to_frame().T
            prediction = float(model.predict(baseline)[0])
            if np.isfinite(prediction):
                predictions[target] = max(0.0, prediction)
        except Exception:
            continue
    return predictions


def build_optimal_launch_recommendation(
    raw_df: pd.DataFrame,
    selected_code: str,
    category: str = "All categories",
    objective: str = "Balanced growth",
) -> dict[str, Any]:
    """Return the next-batch code, historical benchmarks, optional ML estimates, and references."""
    next_code = infer_next_batch_code(selected_code)
    program_frame = _program_level_data(raw_df)
    history = _select_history(program_frame, selected_code, category)

    metric_labels = list(METRIC_CANDIDATES.keys())
    available_labels = [
        label for label in metric_labels
        if f"__metric__{label}" in history.columns and not _metric_series(history, label).empty
    ]
    if history.empty or not available_labels:
        return {
            "next_code": next_code,
            "history_count": len(history),
            "error": "No usable historical metrics were found for this program family.",
            "metrics": {},
            "references": pd.DataFrame(),
            "method": "No estimate",
        }

    # Robust median benchmarks, using the same comparable family first.
    metrics = {
        label: float(_metric_series(history, label).median())
        for label in available_labels
    }

    # Derived metrics are calculated from the benchmark values, not averaged row-level ratios.
    metrics["Estimated paid CPL"] = _safe_ratio(metrics.get("Paid spend", float("nan")), metrics.get("Paid leads", float("nan")))
    metrics["Estimated paid CAC"] = _safe_ratio(metrics.get("Paid spend", float("nan")), metrics.get("Paid enrollments", float("nan")))
    metrics["Lead → application"] = _safe_ratio(metrics.get("Applications", float("nan")), metrics.get("Leads", float("nan")))
    metrics["Application → enrollment"] = _safe_ratio(metrics.get("Enrollments", float("nan")), metrics.get("Applications", float("nan")))

    # Rank historical references by the requested business objective.
    ref = history.copy()
    gcv_col = "__metric__GCV"
    paid_spend_col = "__metric__Paid spend"
    paid_enroll_col = "__metric__Paid enrollments"
    if gcv_col in ref.columns:
        ref["__rank_gcv"] = ref[gcv_col]
    if paid_spend_col in ref.columns and paid_enroll_col in ref.columns:
        ref["__rank_cac"] = ref[paid_spend_col] / ref[paid_enroll_col].where(ref[paid_enroll_col] > 0)

    if objective == "Maximize GCV" and "__rank_gcv" in ref.columns:
        ref = ref.sort_values("__rank_gcv", ascending=False, na_position="last")
    elif objective == "Minimize CAC" and "__rank_cac" in ref.columns:
        ref = ref.sort_values("__rank_cac", ascending=True, na_position="last")
    elif "__rank_gcv" in ref.columns and "__rank_cac" in ref.columns:
        ref["__gcv_rank"] = ref["__rank_gcv"].rank(pct=True)
        ref["__cac_rank"] = ref["__rank_cac"].rank(pct=True, ascending=False)
        ref["__balanced_rank"] = ref[["__gcv_rank", "__cac_rank"]].mean(axis=1)
        ref = ref.sort_values("__balanced_rank", ascending=False, na_position="last")

    references = ref.head(5).copy()
    category_col = _find_column(references, ["category", "program_category"])
    reference_cols = ["program_code"]
    if category_col:
        reference_cols.append(category_col)
    for label, candidates in METRIC_CANDIDATES.items():
        original = _find_column(references, candidates)
        if original and original not in reference_cols:
            reference_cols.append(original)
    references = references[reference_cols].rename(columns={category_col: "category"} if category_col else {})

    # ML only provides supplemental estimates when there are enough distinct batches.
    ml = _try_ml_predictions(history, ["Leads", "Applications", "Enrollments", "GCV"])
    if ml:
        for label, value in ml.items():
            metrics[f"ML estimate: {label}"] = value
    method = "Historical benchmark"
    if ml:
        method = "Historical benchmark + ML estimates"

    return {
        "next_code": next_code,
        "history_count": len(history),
        "metrics": metrics,
        "references": references,
        "method": method,
        "objective": objective,
        "family_matched": bool(_family_prefix(selected_code)),
    }


def render_optimal_launch_metrics(st, raw_df: pd.DataFrame, selected_code: str) -> None:
    """Render this feature in the existing app position; no app theme changes."""
    with st.container(border=True):
        st.markdown("### Optimal Launch Metrics")

        if not selected_code:
            st.caption("Select a New Batch above to automatically identify the next batch and estimate its launch targets.")
            return

        next_code = infer_next_batch_code(selected_code)
        if not next_code:
            st.warning("The selected batch code does not end in a number, so the next batch code cannot be inferred automatically.")
            return

        st.markdown(f"**Predicted next launch:** `{next_code}`")
        st.caption(
            "Uses comparable historical batches first. ML estimates are shown only when enough independent historical observations are available."
        )

        categories = sorted(
            str(value) for value in raw_df.get("category", pd.Series(dtype=str)).dropna().unique()
            if str(value).strip()
        )
        col1, col2 = st.columns([1.2, 1.5])
        with col1:
            category = st.selectbox(
                "Comparable category",
                ["All categories"] + categories,
                key="optimal_launch_category",
            )
        with col2:
            objective = st.selectbox(
                "Optimization goal",
                ["Balanced growth", "Maximize GCV", "Minimize CAC"],
                key="optimal_launch_goal",
            )

        if st.button("✦ Get Optimal Metrics", key="get_optimal_launch_metrics", type="primary"):
            with st.spinner("Evaluating historical batches and launch targets..."):
                result = build_optimal_launch_recommendation(raw_df, selected_code, category, objective)

            if result.get("error"):
                st.warning(result["error"])
                return

            st.markdown(f"#### Recommended benchmark for `{result['next_code']}`")
            st.caption(
                f"Method: {result['method']} · Historical comparable batches: {result['history_count']} · Objective: {objective}"
            )

            metrics = result["metrics"]
            display_order = [
                "Program fee", "Total spend", "GCV", "Leads", "Warm leads",
                "Applications", "Enrollments", "Paid spend", "Paid leads",
                "Paid enrollments", "Paid GCV", "Estimated paid CPL", "Estimated paid CAC",
                "Lead → application", "Application → enrollment",
                "ML estimate: Leads", "ML estimate: Applications",
                "ML estimate: Enrollments", "ML estimate: GCV",
            ]
            items = [(label, metrics[label]) for label in display_order if label in metrics]
            cards = st.columns(4)
            for i, (label, value) in enumerate(items):
                with cards[i % 4]:
                    if pd.isna(value):
                        display = "N/A"
                    elif label in {"Lead → application", "Application → enrollment"}:
                        display = f"{value * 100:,.1f}%"
                    elif label in MONEY_METRICS or label.startswith("ML estimate: GCV"):
                        display = f"₹{value:,.0f}"
                    else:
                        display = f"{value:,.0f}"
                    st.metric(label, display)

            refs = result["references"]
            if not refs.empty:
                st.markdown("#### Historical batches used as references")
                st.dataframe(refs, use_container_width=True, hide_index=True)

            if result["method"] == "Historical benchmark":
                st.info(
                    "This recommendation uses robust historical benchmarks because the available comparable history "
                    "is not sufficient for a reliable ML estimate. It is a planning baseline, not a validated forecast."
                )
            else:
                st.info(
                    "ML estimates are supplemental model-assisted benchmarks, not guaranteed outcomes. "
                    "Validate against a time-based holdout and compare actual results after launch before relying on them for budget commitments."
                )
