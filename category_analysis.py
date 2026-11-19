"""AI-led category intelligence: hidden patterns, fair comparisons and opportunities.

Python computes all metrics; the LLM only interprets verified findings.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd


# Source-level additive measures: aggregate across source_created to one program.
ADDITIVE = [
    "leads", "warm_leads", "applications", "enrollments", "gsv_cr", "spends",
    "Planned_Budget", "Planned_Lead", "Planned_Application", "Planned_Enrollment",
    "Planned_GSV", "paid_leads", "paid_enrollments", "paid_gsv_cr", "paid_spends",
    "planned_paid_leads", "planned_paid_enrollments", "planned_paid_gsv", "planned_paid_spends",
    "previous_batch_leads", "previous_batch_warm_leads", "previous_batch_applications",
    "previous_batch_enrollments", "previous_batch_gsv_cr", "previous_batch_spends",
    "previous_batch_paid_leads", "previous_batch_paid_enrollments",
    "previous_batch_paid_gsv_cr", "previous_batch_paid_spends",
]
ATTRS = [
    "tech_management", "category", "financial_year", "program_status_f", "institute_name_new",
    "New_Repeat", "business_line", "program_zone", "program_tier", "program_name",
    "program_fee_bucket", "sale_duration_in_days_new", "previous_batch_code",
    "program_relevance", "MBA_non_MBA", "Degree_Non_Degree", "program_class", "free_paid",
    "sales_start_date", "program_code", "program_fee", "total_program_fees_no_gst",
]


def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace(",", "", regex=False)
                         .str.replace("₹", "", regex=False).str.replace("%", "", regex=False), errors="coerce")


def prepare_programs(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One row per program. Source metrics are summed; stable program attributes kept once."""
    if raw is None or raw.empty:
        return pd.DataFrame(), {"raw_rows": 0, "programs": 0}
    d = raw.copy()
    if "program_code" not in d or "tech_management" not in d:
        return pd.DataFrame(), {"raw_rows": len(d), "programs": 0,
                                  "missing_columns": [c for c in ["program_code", "tech_management"] if c not in d]}
    d["program_code"] = d["program_code"].astype("string").str.strip()
    d = d[d.program_code.notna() & d.program_code.ne("")].copy()
    metric_cols = [c for c in ADDITIVE if c in d]
    for c in metric_cols:
        d[c] = _numeric(d[c])
    # Keep first non-null program attribute. The uploaded data was checked: each of
    # these attributes has at most one distinct non-null value per program_code.
    agg: dict[str, Any] = {c: "first" for c in ATTRS if c in d}
    agg.update({c: (lambda x: x.sum(min_count=1)) for c in metric_cols})
    p = d.groupby("program_code", as_index=False, dropna=False).agg(agg)
    for c in [x for x in ATTRS if x in p and x not in metric_cols]:
        p[c] = p[c].astype("string").str.strip().replace("", pd.NA)
    if "tech_management" in p:
        p["tech_management"] = p.tech_management.fillna("Unknown")
    if "financial_year" in p:
        p["financial_year"] = p.financial_year.fillna("Unknown")
    for n, den, label in [("warm_leads", "leads", "L2W"), ("applications", "leads", "L2A"),
                          ("applications", "warm_leads", "W2A"), ("enrollments", "applications", "A2E"),
                          ("enrollments", "leads", "L2E")]:
        if n in p and den in p:
            p[label] = np.where(p[den] > 0, p[n] / p[den], np.nan)
    if "paid_enrollments" in p and "paid_leads" in p:
        p["Paid L2E"] = np.where(p.paid_leads > 0, p.paid_enrollments / p.paid_leads, np.nan)
    if "gsv_cr" in p and "enrollments" in p:
        p["GSV per enrollment"] = np.where(p.enrollments > 0, p.gsv_cr / p.enrollments, np.nan)
    # Program fee canonical numeric feature.
    for c in ["program_fee", "total_program_fees_no_gst", "sale_duration_in_days_new"]:
        if c in p:
            p[c] = _numeric(p[c])
    # Use robust fallback when program_fee is absent.
    if "program_fee" not in p or p.program_fee.notna().sum() == 0:
        if "total_program_fees_no_gst" in p:
            p["program_fee"] = p.total_program_fees_no_gst
    meta = {
        "raw_rows": int(len(d)), "programs": int(p.program_code.nunique()),
        "source_count": int(d.source_created.nunique()) if "source_created" in d else None,
        "source_row_metrics_coverage": {c: round(float(d[c].notna().mean()), 4) for c in metric_cols},
        "attribute_consistency_checked": True,
    }
    return p, meta


def _rate(num: pd.Series, den: pd.Series) -> float:
    a, b = num.sum(min_count=1), den.sum(min_count=1)
    return float(a / b) if pd.notna(a) and pd.notna(b) and b > 0 else np.nan


def _fmt_pct(x: float) -> str:
    return "N/A" if pd.isna(x) else f"{x:.1%}"


def _segment_key(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ["New_Repeat", "program_fee_bucket"] if c in df]
    if not cols:
        return pd.Series("All", index=df.index)
    return df[cols].fillna("Unknown").astype(str).agg(" | ".join, axis=1)


def adjusted_category_comparison(p: pd.DataFrame, min_programs: int = 8) -> pd.DataFrame:
    """Direct-standardize each category's segment L2E to the same overall segment mix."""
    needed = {"tech_management", "leads", "enrollments"}
    if not needed.issubset(p.columns):
        return pd.DataFrame()
    d = p.copy()
    d["_segment"] = _segment_key(d)
    # Common reference weights: overall lead distribution across New/Repeat x fee bucket.
    overall = d.groupby("_segment").agg(
        programs=("program_code", "nunique"), leads=("leads", "sum"),
        enrollments=("enrollments", "sum")
    )
    total_leads = overall.leads.sum()
    if pd.isna(total_leads) or total_leads <= 0:
        return pd.DataFrame()
    weights = (overall.leads / total_leads).to_dict()
    rows = []
    for cat, g in d.groupby("tech_management"):
        actual = _rate(g.enrollments, g.leads)
        seg_rates = {segment: _rate(sg.enrollments, sg.leads)
                     for segment, sg in g.groupby("_segment")}
        weighted_sum, used_weight = 0.0, 0.0
        for segment, weight in weights.items():
            rate = seg_rates.get(segment, np.nan)
            if pd.notna(rate):
                weighted_sum += float(weight) * float(rate)
                used_weight += float(weight)
        adjusted = weighted_sum / used_weight if used_weight > 0 else np.nan
        rows.append({
            "tech_management": cat, "programs": int(g.program_code.nunique()),
            "leads": float(g.leads.sum(min_count=1)),
            "enrollments": float(g.enrollments.sum(min_count=1)),
            "raw_l2e": actual, "mix_adjusted_l2e": adjusted,
            "pp_difference": (adjusted - actual) * 100 if pd.notna(adjusted) and pd.notna(actual) else np.nan,
            "adjustment_coverage": used_weight,
        })
    out = pd.DataFrame(rows)
    return out[out.programs >= min_programs].sort_values("mix_adjusted_l2e", ascending=False)


def _find_mix_reversal(p: pd.DataFrame, findings: list[dict]) -> pd.DataFrame:
    comp = adjusted_category_comparison(p)
    if len(comp) >= 2:
        comp["raw_rank"] = comp.raw_l2e.rank(ascending=False, method="min")
        comp["adjusted_rank"] = comp.mix_adjusted_l2e.rank(ascending=False, method="min")
        comp["rank_shift"] = comp.raw_rank - comp.adjusted_rank
        shifted = comp[comp.rank_shift.abs() >= 1]
        if not shifted.empty:
            top_raw = comp.sort_values("raw_l2e", ascending=False).iloc[0]
            top_adj = comp.sort_values("mix_adjusted_l2e", ascending=False).iloc[0]
            if top_raw.tech_management != top_adj.tech_management:
                findings.append({"type": "mix_effect", "priority": "High",
                    "title": "Category leadership changes after adjusting for program mix",
                    "evidence": f"Raw L2E leader: {top_raw.tech_management} ({_fmt_pct(top_raw.raw_l2e)}); "
                                f"mix-adjusted leader: {top_adj.tech_management} ({_fmt_pct(top_adj.mix_adjusted_l2e)}).",
                    "interpretation": "The headline category ranking is partly influenced by its New/Repeat and fee-bucket composition.",
                    "caveat": "Adjustment is observational and only controls for available New/Repeat and fee-bucket dimensions."})
    return comp


def _discover_segment_gaps(p: pd.DataFrame, findings: list[dict], min_n: int = 8) -> pd.DataFrame:
    dims = [c for c in ["tech_management", "New_Repeat", "program_fee_bucket", "business_line", "program_zone", "program_tier"] if c in p]
    rows = []
    if not {"leads", "enrollments"}.issubset(p.columns):
        return pd.DataFrame()
    for dim in dims:
        for val, g in p.groupby(dim, dropna=False):
            n = g.program_code.nunique()
            if n < min_n:
                continue
            rate = _rate(g.enrollments, g.leads)
            peer = p.loc[~p.index.isin(g.index)]
            peer_rate = _rate(peer.enrollments, peer.leads) if len(peer) else np.nan
            if pd.notna(rate) and pd.notna(peer_rate):
                rows.append({"dimension": dim, "segment": str(val), "programs": int(n), "leads": float(g.leads.sum()),
                             "enrollments": float(g.enrollments.sum()), "l2e": rate, "rest_l2e": peer_rate,
                             "difference_pp": (rate - peer_rate) * 100})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["abs_gap"] = result.difference_pp.abs()
        result = result.sort_values("abs_gap", ascending=False)
        for _, r in result.head(5).iterrows():
            direction = "outperforms" if r.difference_pp > 0 else "underperforms"
            findings.append({"type": "segment_pattern", "priority": "High" if abs(r.difference_pp) >= 3 else "Medium",
                "title": f"{r.segment} {direction} the rest on enrollment conversion",
                "evidence": f"{r.dimension}={r.segment}: {r.programs} programs, {r.leads:,.0f} leads, "
                            f"L2E {_fmt_pct(r.l2e)} versus {_fmt_pct(r.rest_l2e)} for all other records "
                            f"({r.difference_pp:+.1f} percentage points).",
                "interpretation": "This segment is a candidate for a deeper driver investigation; it is not proof of causality.",
                "caveat": "Segments are tested independently and may overlap; comparison is unadjusted."})
    return result


def _discover_correlations(p: pd.DataFrame, findings: list[dict]) -> pd.DataFrame:
    candidates = [c for c in ["program_fee", "sale_duration_in_days_new", "sales_period_completed_percentage",
                              "sales_ageing", "leads", "paid_leads", "spends", "paid_spends"] if c in p]
    outcomes = [c for c in ["L2E", "L2A", "A2E", "GSV per enrollment", "Paid L2E"] if c in p]
    rows = []
    for x in candidates:
        for y in outcomes:
            z = p[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(z) < 25 or z[x].nunique() < 5 or z[y].nunique() < 5:
                continue
            corr = z[x].corr(z[y], method="spearman")
            if pd.notna(corr):
                rows.append({"feature": x, "outcome": y, "spearman_rho": float(corr), "programs": int(len(z))})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_rho"] = out.spearman_rho.abs()
        out = out.sort_values("abs_rho", ascending=False)
        for _, r in out.head(5).iterrows():
            strength = "strong" if abs(r.spearman_rho) >= .5 else "moderate" if abs(r.spearman_rho) >= .3 else "weak"
            direction = "positive" if r.spearman_rho > 0 else "negative"
            findings.append({"type": "association", "priority": "Medium" if abs(r.spearman_rho) >= .3 else "Low",
                "title": f"{strength.title()} {direction} association: {r.feature} and {r.outcome}",
                "evidence": f"Spearman rho={r.spearman_rho:+.2f} across {r.programs} programs with both values available.",
                "interpretation": f"Higher {r.feature} tends to be associated with {'higher' if r.spearman_rho > 0 else 'lower'} {r.outcome}.",
                "caveat": "Correlation is not causation and may reflect category, fee, institute, or program-mix effects."})
    return out


def _archetypes(p: pd.DataFrame, findings: list[dict]) -> tuple[pd.DataFrame, str]:
    """Cluster programs and assign readable, data-derived exploratory profiles."""
    cols = [c for c in ["L2W", "L2A", "A2E", "Paid L2E", "GSV per enrollment", "sale_duration_in_days_new"] if c in p]
    if len(cols) < 3 or len(p) < 30:
        return pd.DataFrame(), "Not enough program-level metrics for clustering."
    try:
        from sklearn.cluster import KMeans
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        X = p[cols].replace([np.inf, -np.inf], np.nan).copy()
        for c in cols:
            lo, hi = X[c].quantile(.02), X[c].quantile(.98)
            if pd.notna(lo) and pd.notna(hi) and hi > lo:
                X[c] = X[c].clip(lo, hi)
        X = SimpleImputer(strategy="median").fit_transform(X)
        X = StandardScaler().fit_transform(X)
        k = min(4, max(2, int(round(math.sqrt(len(p) / 30)))))
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        out = p[[c for c in ["program_code", "program_name", "tech_management", "category", "New_Repeat", "leads", "enrollments", "gsv_cr"] if c in p]].copy()
        out["cluster_id"] = labels
        stats = p.assign(cluster_id=labels).groupby("cluster_id").agg(
            programs=("program_code", "nunique"), leads=("leads", "sum"), enrollments=("enrollments", "sum"),
            median_l2e=("L2E", "median") if "L2E" in p else ("program_code", "count"),
            median_l2a=("L2A", "median") if "L2A" in p else ("program_code", "count"),
            median_a2e=("A2E", "median") if "A2E" in p else ("program_code", "count"),
            median_paid_l2e=("Paid L2E", "median") if "Paid L2E" in p else ("program_code", "count"),
            median_gsv_per_enrollment=("GSV per enrollment", "median") if "GSV per enrollment" in p else ("program_code", "count"),
            median_leads=("leads", "median"),
        ).reset_index()
        labels_by_cluster = {}
        for _, r in stats.iterrows():
            if pd.notna(r.median_l2e) and pd.notna(r.median_gsv_per_enrollment) and r.median_l2e >= stats.median_l2e.median() and r.median_gsv_per_enrollment >= stats.median_gsv_per_enrollment.median():
                profile = "High-conversion / high-value"
            elif pd.notna(r.median_leads) and pd.notna(r.median_l2e) and r.median_leads >= stats.median_leads.median() and r.median_l2e < stats.median_l2e.median():
                profile = "High-volume / low-conversion"
            elif pd.notna(r.median_l2a) and pd.notna(r.median_a2e) and r.median_l2a < stats.median_l2a.median() and r.median_a2e >= stats.median_a2e.median():
                profile = "Application bottleneck"
            elif pd.notna(r.median_paid_l2e) and pd.notna(r.median_l2e) and r.median_paid_l2e > r.median_l2e:
                profile = "Paid-channel efficient"
            else:
                profile = "Mixed performance profile"
            labels_by_cluster[int(r.cluster_id)] = profile
        out["archetype"] = out.cluster_id.map(labels_by_cluster)
        stats["archetype"] = stats.cluster_id.map(labels_by_cluster)
        if len(X) > k + 1 and len(np.unique(labels)) > 1:
            sil = silhouette_score(X, labels)
        else:
            sil = np.nan
        for _, r in stats.iterrows():
            findings.append({"type": "archetype", "priority": "Medium",
                "title": f"{r.archetype}: {int(r.programs)} programs share a similar performance profile",
                "evidence": f"{r.leads:,.0f} leads and {r.enrollments:,.0f} enrollments; median L2E {_fmt_pct(r.median_l2e)}.",
                "interpretation": "Use the program drill-down to inspect which categories and institutes are concentrated in this profile.",
                "caveat": "Clusters are exploratory groupings based on standardized conversion, value and sales-duration features; archetype labels are heuristic summaries."})
        # Attach cluster profile statistics to each program row for a useful drill-down.
        out = out.merge(stats[["cluster_id", "programs", "median_l2e", "median_l2a", "median_a2e", "median_paid_l2e", "median_gsv_per_enrollment"]], on="cluster_id", how="left")
        note = f"KMeans with {k} clusters; silhouette={sil:.2f}" if pd.notna(sil) else f"KMeans with {k} clusters; silhouette unavailable"
        return out, note
    except Exception as exc:
        return pd.DataFrame(), f"Clustering unavailable: {exc}"


def _anomaly_detection(p: pd.DataFrame, findings: list[dict]) -> pd.DataFrame:
    """Flag unusual multivariate program profiles for human review, not as errors."""
    cols = [c for c in ["L2W", "L2A", "A2E", "Paid L2E", "GSV per enrollment", "sale_duration_in_days_new", "leads"] if c in p]
    if len(cols) < 3 or len(p) < 40:
        return pd.DataFrame()
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import RobustScaler
        X = p[cols].replace([np.inf, -np.inf], np.nan).copy()
        for c in cols:
            lo, hi = X[c].quantile(.01), X[c].quantile(.99)
            if pd.notna(lo) and pd.notna(hi) and hi > lo:
                X[c] = X[c].clip(lo, hi)
        X = SimpleImputer(strategy="median").fit_transform(X)
        X = RobustScaler().fit_transform(X)
        model = IsolationForest(n_estimators=250, contamination=0.02, random_state=42)
        flags = model.fit_predict(X)
        scores = -model.score_samples(X)
        out = p[[c for c in ["program_code", "program_name", "tech_management", "category", "New_Repeat", "leads", "enrollments", "L2E", "gsv_cr"] if c in p]].copy()
        out["anomaly_score"] = scores
        out["review_flag"] = flags == -1
        out = out[out.review_flag].sort_values("anomaly_score", ascending=False)
        if not out.empty:
            findings.append({"type": "anomaly", "priority": "Medium",
                "title": f"{len(out)} programs have unusual multi-metric profiles",
                "evidence": f"Isolation Forest flagged approximately {len(out) / len(p):.1%} of programs for review across {', '.join(cols)}.",
                "interpretation": "Inspect the flagged programs for unusual funnel behavior, data issues, or genuinely distinctive commercial models.",
                "caveat": "Anomaly flags mean statistically unusual, not necessarily poor performance or incorrect data. The method flags about 2% by design."})
        return out
    except Exception as exc:
        return pd.DataFrame()


def _batch_changes(p: pd.DataFrame, findings: list[dict]) -> pd.DataFrame:
    pairs = [("leads", "previous_batch_leads", "Leads"), ("applications", "previous_batch_applications", "Applications"),
             ("enrollments", "previous_batch_enrollments", "Enrollments"), ("gsv_cr", "previous_batch_gsv_cr", "GSV")]
    rows = []
    if "previous_batch_code" not in p:
        return pd.DataFrame()
    matched = p[p.previous_batch_code.notna() & p.previous_batch_code.astype(str).str.strip().ne("")].copy()
    for cat, g in matched.groupby("tech_management"):
        row = {"tech_management": cat, "matched_programs": int(g.program_code.nunique())}
        for cur, prev, label in pairs:
            if cur in g and prev in g:
                a, b = g[cur].sum(min_count=1), g[prev].sum(min_count=1)
                row[f"{label} current"] = a
                row[f"{label} previous"] = b
                row[f"{label} change_pct"] = (a / b - 1) * 100 if pd.notna(a) and pd.notna(b) and b > 0 else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        for _, r in out.iterrows():
            change = r.get("Enrollments change_pct", np.nan)
            if pd.notna(change) and abs(change) >= 15 and r.matched_programs >= 5:
                findings.append({"type": "batch_change", "priority": "High" if change < 0 else "Medium",
                    "title": f"{r.tech_management}: matched-program enrollments changed {change:+.1f}%",
                    "evidence": f"{int(r.matched_programs)} programs with previous-batch references; current enrollments "
                                f"{r.get('Enrollments current', np.nan):,.0f} vs previous-batch values {r.get('Enrollments previous', np.nan):,.0f}.",
                    "interpretation": "Investigate whether changes in lead volume, conversion, or program value explain the movement.",
                    "caveat": "This compares current records with their embedded previous-batch metrics; only programs with a previous_batch_code are included."})
    return out


def _source_dependency(raw: pd.DataFrame, p: pd.DataFrame, findings: list[dict]) -> pd.DataFrame:
    if not {"source_created", "program_code", "leads"}.issubset(raw.columns):
        return pd.DataFrame()
    d = raw.copy()
    for c in ["leads", "applications", "enrollments", "gsv_cr"]:
        if c in d:
            d[c] = _numeric(d[c])
    # Map each program to its canonical tech_management.
    d = d.merge(p[["program_code", "tech_management"]], on="program_code", how="left", suffixes=("", "_program"))
    rows = []
    for (cat, src), g in d.groupby(["tech_management", "source_created"], dropna=False):
        leads = g.leads.sum(min_count=1)
        apps = g.applications.sum(min_count=1) if "applications" in g else np.nan
        enr = g.enrollments.sum(min_count=1) if "enrollments" in g else np.nan
        rows.append({"tech_management": cat, "source_created": src, "source_rows": len(g),
                     "programs": g.program_code.nunique(), "leads": leads, "applications": apps,
                     "enrollments": enr, "L2A": apps / leads if pd.notna(apps) and pd.notna(leads) and leads > 0 else np.nan,
                     "L2E": enr / leads if pd.notna(enr) and pd.notna(leads) and leads > 0 else np.nan})
    out = pd.DataFrame(rows)
    if not out.empty:
        for cat, g in out.groupby("tech_management"):
            total = g.leads.sum(min_count=1)
            top = g.sort_values("leads", ascending=False).iloc[0]
            share = top.leads / total if pd.notna(total) and total > 0 else np.nan
            if pd.notna(share) and share >= .55:
                findings.append({"type": "source_concentration", "priority": "Medium",
                    "title": f"{cat} is highly dependent on {top.source_created}",
                    "evidence": f"{top.source_created} contributes {share:.1%} of the category's leads.",
                    "interpretation": "A large share from one channel may create acquisition concentration risk; compare channel quality and resilience.",
                    "caveat": "Source rows represent source attribution; channel-level spend is not available consistently enough for a complete ROI conclusion."})
    return out


def discover_patterns(raw: pd.DataFrame, min_programs: int = 8) -> dict:
    p, meta = prepare_programs(raw)
    if p.empty:
        return {"programs": p, "meta": meta, "findings": [], "tables": {}}
    findings: list[dict] = []
    mix = _find_mix_reversal(p, findings)
    segments = _discover_segment_gaps(p, findings, min_programs)
    correlations = _discover_correlations(p, findings)
    archetypes, cluster_note = _archetypes(p, findings)
    anomalies = _anomaly_detection(p, findings)
    batches = _batch_changes(p, findings)
    sources = _source_dependency(raw, p, findings)
    # Opportunity scenario: use category/segment peer rates, not the global average.
    opp_rows = []
    if {"tech_management", "leads", "enrollments", "New_Repeat", "program_fee_bucket"}.issubset(p.columns):
        p2 = p.copy(); p2["_segment"] = _segment_key(p2)
        seg_rates = p2.groupby("_segment").apply(lambda g: _rate(g.enrollments, g.leads), include_groups=False)
        for cat, g in p2.groupby("tech_management"):
            gap = 0.0; used = 0; leads = g.leads.sum(min_count=1)
            for s, sg in g.groupby("_segment"):
                peers = p2[(p2._segment == s) & (p2.tech_management != cat)]
                peer_rate = _rate(peers.enrollments, peers.leads) if peers.program_code.nunique() >= min_programs else np.nan
                own_rate = _rate(sg.enrollments, sg.leads)
                if pd.notna(peer_rate) and pd.notna(own_rate) and peer_rate > own_rate and pd.notna(sg.leads.sum()):
                    gap += float(sg.leads.sum()) * (peer_rate - own_rate); used += int(sg.program_code.nunique())
            opp_rows.append({"tech_management": cat, "programs_in_segments_with_peer_data": used,
                             "estimated_enrollment_opportunity": max(0, gap) if used else np.nan,
                             "lead_coverage": leads})
    opportunities = pd.DataFrame(opp_rows)
    if not opportunities.empty:
        for _, r in opportunities.dropna(subset=["estimated_enrollment_opportunity"]).sort_values("estimated_enrollment_opportunity", ascending=False).head(3).iterrows():
            if r.estimated_enrollment_opportunity >= 5:
                findings.append({"type": "opportunity", "priority": "High",
                    "title": f"Potential enrollment opportunity in {r.tech_management}",
                    "evidence": f"Approximately {r.estimated_enrollment_opportunity:,.1f} additional enrollments in matched segments if the category reached peer-category conversion rates.",
                    "interpretation": "Prioritize the underlying New/Repeat and fee-bucket segments with the largest conversion gaps.",
                    "caveat": "Scenario estimate, not a forecast. It uses other categories' pooled L2E in the same New/Repeat x fee-bucket segment and only includes segments with sufficient peer volume."})
    priority = {"High": 0, "Medium": 1, "Low": 2}
    findings.sort(key=lambda x: (priority.get(x["priority"], 3), x["type"]))
    return {"programs": p, "meta": meta, "findings": findings,
            "tables": {"mix_adjusted": mix, "segment_gaps": segments, "associations": correlations,
                       "archetypes": archetypes, "anomalies": anomalies, "batch_changes": batches, "source_dependency": sources,
                       "opportunities": opportunities}, "cluster_note": cluster_note}


def _llm_commentary(st, findings: list[dict], question: str = "") -> str:
    """Interpret precomputed findings only. Never invent or recalculate metrics."""
    from openai import OpenAI
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured in Streamlit Secrets.")
    model = st.secrets.get("OPENAI_MODEL", "gpt-5.6")
    payload = json.dumps(findings[:18], ensure_ascii=False, default=str)
    prompt = f"""User investigation question: {question or 'Identify the most important non-obvious category patterns.'}

Verified Python findings (treat as the only numerical evidence):
{payload}

Write an executive investigation brief with: (1) strongest supported discoveries, (2) plausible explanations clearly labelled as hypotheses, (3) what to validate next, and (4) practical actions. Do not invent numbers, causal claims, or findings. Mention caveats and small samples. If findings are weak, say so."""
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a rigorous commercial analytics investigator. Python owns all calculations; you interpret only supplied evidence."},
                  {"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content or "No interpretation returned."


def render_category_analysis(st, raw_df: pd.DataFrame) -> None:
    st.markdown("## AI Category Intelligence")
    st.caption("An investigation engine that discovers non-obvious patterns across categories, program mix, sources and previous batches.")
    @st.cache_data(show_spinner="Scanning program patterns...")
    def _cached_discovery(dataframe: pd.DataFrame):
        return discover_patterns(dataframe)

    result = _cached_discovery(raw_df)
    programs, meta, findings, tables = result["programs"], result["meta"], result["findings"], result["tables"]
    if programs.empty:
        st.error("Cannot run category intelligence. Required columns: program_code and tech_management.")
        return

    a, b, c, d = st.columns(4)
    a.metric("Programs analysed", f"{meta['programs']:,}")
    b.metric("Source rows consolidated", f"{meta['raw_rows']:,}")
    c.metric("Categories", f"{programs.tech_management.nunique():,}")
    d.metric("Patterns detected", f"{len(findings):,}")
    st.caption("All actual source metrics are consolidated to one row per program before program-level analysis. Findings are associations or scenarios, not causal proof.")

    with st.expander("Investigation scope", expanded=False):
        x1, x2, x3 = st.columns(3)
        with x1:
            categories = sorted(programs.tech_management.dropna().astype(str).unique())
            selected_categories = st.multiselect("Tech management", categories, default=categories, key="intel_categories")
        with x2:
            fys = sorted(programs.financial_year.dropna().astype(str).unique()) if "financial_year" in programs else []
            selected_fys = st.multiselect("Financial year", fys, default=fys, key="intel_fys")
        with x3:
            new_repeat = sorted(programs.New_Repeat.dropna().astype(str).unique()) if "New_Repeat" in programs else []
            selected_nr = st.multiselect("New / Repeat", new_repeat, default=new_repeat, key="intel_nr")
    filtered = programs.copy()
    for col, vals in [("tech_management", selected_categories), ("financial_year", selected_fys), ("New_Repeat", selected_nr)]:
        if col in filtered:
            filtered = filtered[filtered[col].astype(str).isin(vals)] if vals else filtered.iloc[0:0]
    if len(filtered) != len(programs):
        # Re-run all methods against the filtered population, while retaining raw rows for source analysis.
        raw_filtered = raw_df[raw_df.program_code.astype(str).isin(filtered.program_code.astype(str))]
        result = _cached_discovery(raw_filtered)
        programs, meta, findings, tables = result["programs"], result["meta"], result["findings"], result["tables"]
    if not findings:
        st.info("No sufficiently strong patterns met the current minimum sample-size rules. Try widening the investigation scope.")

    st.markdown("### Discoveries")
    if findings:
        for i, f in enumerate(findings[:15], 1):
            with st.container(border=True):
                head = st.columns([5, 1])
                head[0].markdown(f"**{i}. {f['title']}**")
                head[1].markdown(f"`{f['priority']} priority`")
                st.write(f["evidence"])
                st.caption(f"Interpretation: {f['interpretation']}")
                st.caption(f"Caveat: {f['caveat']}")
    else:
        st.write("The discovery scan did not find a pattern strong enough to surface.")

    st.markdown("### Ask an investigation question")
    question = st.text_input("What should the AI focus on?", placeholder="e.g. Explain the category mix-adjusted conversion findings")
    st.caption("The discovery scan calculates the evidence automatically. The optional LLM brief prioritizes and explains those verified findings; it does not run arbitrary new calculations from the question.")
    if st.button("Generate AI investigation brief", type="primary", disabled=not findings, key="run_category_ai_brief"):
        with st.spinner("Interpreting verified analytical findings..."):
            try:
                st.markdown(_llm_commentary(st, findings, question))
            except Exception as exc:
                st.warning(f"LLM interpretation unavailable: {exc}")
                st.info("The deterministic discovery results above remain available.")

    tabs = st.tabs(["Fair comparison", "Program archetypes", "Anomaly review", "Batch intelligence", "Source dependency", "Opportunity scenarios", "Associations"])
    with tabs[0]:
        st.markdown("**Raw vs mix-adjusted L2E**")
        t = tables.get("mix_adjusted", pd.DataFrame())
        if not t.empty:
            display = t.copy()
            for col in ["raw_l2e", "mix_adjusted_l2e", "adjustment_coverage"]:
                display[col] = display[col].map(_fmt_pct)
            display["pp_difference"] = display["pp_difference"].map(lambda x: f"{x:+.1f} pp" if pd.notna(x) else "N/A")
            st.dataframe(display, use_container_width=True, hide_index=True)
            st.caption("Direct standardization by New/Repeat and fee bucket. This is a diagnostic adjustment, not a causal model.")
        else:
            st.info("Not enough category/segment coverage for mix-adjusted comparison.")
        seg = tables.get("segment_gaps", pd.DataFrame())
        if not seg.empty:
            st.markdown("**Largest segment differences vs the rest of the dataset**")
            st.dataframe(seg.head(15), use_container_width=True, hide_index=True)
    with tabs[1]:
        st.caption(result.get("cluster_note", ""))
        t = tables.get("archetypes", pd.DataFrame())
        if not t.empty:
            st.dataframe(t.merge(programs[[c for c in ["program_code", "program_name", "tech_management", "category", "New_Repeat", "leads", "enrollments"] if c in programs]], on=[c for c in ["program_code", "program_name", "tech_management", "category", "New_Repeat", "leads", "enrollments"] if c in t and c in programs], how="left") if False else t, use_container_width=True, hide_index=True)
            st.caption("Cluster labels are generated from conversion, value and sales-duration profiles. They are exploratory, not fixed business classifications.")
        else:
            st.info("Clustering could not be run on the available fields.")
    with tabs[2]:
        t = tables.get("anomalies", pd.DataFrame())
        if not t.empty:
            st.dataframe(t, use_container_width=True, hide_index=True)
            st.caption("Anomaly flags identify unusual profiles for review, not necessarily errors or poor performance.")
        else:
            st.info("No anomaly results are available.")
    with tabs[3]:
        t = tables.get("batch_changes", pd.DataFrame())
        if not t.empty:
            st.dataframe(t, use_container_width=True, hide_index=True)
            st.caption("Current program totals are compared with previous-batch metrics embedded for the same program. Only records with previous_batch_code are included.")
        else:
            st.info("No matched previous-batch metrics were available.")
    with tabs[4]:
        t = tables.get("source_dependency", pd.DataFrame())
        if not t.empty:
            st.dataframe(t.sort_values(["tech_management", "leads"], ascending=[True, False]), use_container_width=True, hide_index=True)
            st.caption("Source rows are retained for channel attribution analysis. Spend/ROI claims are not made because source-level spend coverage is incomplete.")
        else:
            st.info("Source attribution fields are unavailable.")
    with tabs[5]:
        t = tables.get("opportunities", pd.DataFrame())
        if not t.empty:
            st.dataframe(t, use_container_width=True, hide_index=True)
            st.caption("Scenario: category segments reach peer-category L2E within the same New/Repeat and fee-bucket segment. Not a forecast or guaranteed uplift.")
        else:
            st.info("Insufficient matched peer segments to estimate opportunity scenarios.")
    with tabs[6]:
        t = tables.get("associations", pd.DataFrame())
        if not t.empty:
            st.dataframe(t, use_container_width=True, hide_index=True)
            st.caption("Spearman rank correlations are exploratory. They do not establish causation and may be confounded by program mix.")
        else:
            st.info("No sufficiently supported correlations were found.")

    st.markdown("### Program-level evidence explorer")
    view_cols = [c for c in ["program_code", "program_name", "tech_management", "category", "financial_year", "New_Repeat",
                              "program_fee_bucket", "sale_duration_in_days_new", "leads", "applications", "enrollments", "L2E", "gsv_cr", "Paid L2E"] if c in programs]
    st.dataframe(programs[view_cols], use_container_width=True, hide_index=True, height=360)

    _render_category_chat(st, raw_df, programs, result)



def _category_chat_context(raw: pd.DataFrame, programs: pd.DataFrame, result: dict) -> str:
    """Build auditable context for category chat: source schema, analysis and records."""
    parts = [
        "CATEGORY INTELLIGENCE CHAT CONTEXT",
        f"Source rows loaded: {len(raw):,}",
        f"Unique programs after source-row consolidation: {programs.program_code.nunique():,}",
        "Primary grouping: tech_management. Program-level metrics are summed across source_created rows; program attributes are retained once.",
        "Raw source columns: " + ", ".join(map(str, raw.columns)),
        "IMPORTANT: Missing values are unknown/unavailable, not zero. Findings are observational, not causal.",
    ]
    findings = result.get("findings", [])
    parts.append("\nVERIFIED DISCOVERY FINDINGS (Python-generated):\n" +
                 json.dumps(findings[:15], ensure_ascii=False, default=str))
    table_blocks = []
    for name, frame in (result.get("tables") or {}).items():
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            table_blocks.append(f"\n{name} (top 25 rows):\n{frame.head(25).to_csv(index=False)}")
    parts.append("\nANALYSIS TABLES:\n" + "\n".join(table_blocks))

    preferred = [
        "program_code", "program_name", "tech_management", "category", "financial_year",
        "program_status_f", "institute_name_new", "New_Repeat", "business_line",
        "program_fee_bucket", "program_fee", "sale_duration_in_days_new",
        "previous_batch_code", "leads", "warm_leads", "applications", "enrollments",
        "gsv_cr", "spends", "paid_leads", "paid_enrollments", "paid_gsv_cr",
        "paid_spends", "L2A", "A2E", "L2E", "GSV per enrollment",
        "previous_batch_enrollments", "previous_batch_gsv_cr",
    ]
    cols = [c for c in preferred if c in programs.columns]
    data_csv = programs[cols].to_csv(index=False)
    max_data_chars = 80000
    if len(data_csv) > max_data_chars:
        # Keep complete CSV records only; never cut a row in the middle.
        lines = data_csv.splitlines(keepends=True)
        kept = [lines[0]]
        used = len(lines[0])
        for line in lines[1:]:
            if used + len(line) > max_data_chars:
                break
            kept.append(line)
            used += len(line)
        data_csv = "".join(kept)
        data_note = (f"\nPROGRAM DATA LIMITATION: {len(kept)-1:,} of {len(programs):,} program records "
                     "are included due to context limits. Do not claim complete row coverage for calculations.")
    else:
        data_note = "\nAll consolidated program rows and listed fields are included."
    parts.append("\nCONSOLIDATED PROGRAM SOURCE DATA (CSV):\n" + data_csv + data_note)

    if "source_created" in raw.columns and "tech_management" in raw.columns:
        src = raw.copy()
        for c in ["leads", "warm_leads", "applications", "enrollments", "gsv_cr"]:
            if c in src:
                src[c] = _numeric(src[c])
        measures = [c for c in ["leads", "warm_leads", "applications", "enrollments", "gsv_cr"] if c in src]
        if measures:
            summary = src.groupby(["tech_management", "source_created"], dropna=False)[measures].sum(min_count=1).reset_index()
            parts.append("\nSOURCE-CHANNEL SUMMARY BY TECH MANAGEMENT:\n" + summary.to_csv(index=False))
    return "\n\n".join(parts)


def _render_category_chat(st, raw: pd.DataFrame, programs: pd.DataFrame, result: dict) -> None:
    """Context-aware chat with persistent session conversation history."""
    st.markdown("---")
    st.markdown("## Ask the Category Intelligence Agent")
    st.caption(
        "Ask follow-up questions about the source CSV, discovered patterns, analytical tables, "
        "program records, or earlier questions in this conversation. The agent receives this "
        "page's current analysis context on every turn."
    )
    history_key = "category_intelligence_chat_history"
    if history_key not in st.session_state:
        st.session_state[history_key] = []
    for item in st.session_state[history_key]:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

    question = st.chat_input(
        "Ask about category patterns, programs, sources, batches or opportunities...",
        key="category_intelligence_chat_input",
    )
    if not question:
        return

    st.session_state[history_key].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    from agent import ask_agent
    history = st.session_state[history_key][:-1][-12:]
    scoped_raw = raw[raw["program_code"].astype(str).isin(programs["program_code"].astype(str))].copy() if "program_code" in raw else raw
    context = _category_chat_context(scoped_raw, programs, result)
    with st.chat_message("assistant"):
        with st.spinner("Investigating the source data and previous analysis..."):
            try:
                answer = ask_agent(
                    question=question, df=programs, program_code=None,
                    old_batch_code=None, check_df=None, chat_history=history,
                    extra_context=context,
                )
            except Exception as exc:
                answer = (f"I couldn't complete that investigation: `{exc}`\n\n"
                          "Please check that OPENAI_API_KEY and OPENAI_MODEL are configured. "
                          "The analysis above remains available.")
        st.markdown(answer)
    st.session_state[history_key].append({"role": "assistant", "content": answer})
    st.session_state[history_key] = st.session_state[history_key][-24:]
