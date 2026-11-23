"""AI-led category intelligence: hidden patterns, fair comparisons and opportunities.

Python computes all metrics; the LLM only interprets verified findings.
"""
from __future__ import annotations

import json
import math
import html
import re
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
    # Paid metrics and planned-paid fields are program-level snapshots repeated on
    # each source row. Retain one non-null value; never sum these across sources.
    snapshot_cols = [c for c in [
        "paid_leads", "paid_enrollments", "paid_gsv_cr", "paid_spends",
        "planned_paid_leads", "planned_paid_enrollments", "planned_paid_gsv", "planned_paid_spends",
    ] if c in d]
    agg: dict[str, Any] = {c: "first" for c in ATTRS if c in d}
    agg.update({c: (lambda x: x.sum(min_count=1)) for c in metric_cols})
    agg.update({c: (lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan) for c in snapshot_cols})
    p = d.groupby("program_code", as_index=False, dropna=False).agg(agg)
    for c in snapshot_cols:
        p[c] = _numeric(p[c])
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



def _safe_ratio(num, den):
    return np.where(pd.notna(den) & (den > 0) & pd.notna(num), num / den, np.nan)


def _build_management_tables(programs: pd.DataFrame) -> dict:
    """Create category and individual program-batch evidence tables."""
    p = programs.copy()
    for c in ["leads", "warm_leads", "applications", "enrollments", "gsv_cr",
              "spends", "Planned_Budget", "Planned_Lead", "Planned_Application",
              "Planned_Enrollment", "Planned_GSV"]:
        if c in p:
            p[c] = _numeric(p[c])
    for n, den, label in [
        ("warm_leads", "leads", "L2W"), ("applications", "leads", "L2A"),
        ("applications", "warm_leads", "W2A"), ("enrollments", "applications", "A2E"),
        ("enrollments", "leads", "L2E"),
    ]:
        if n in p and den in p:
            p[label] = _safe_ratio(p[n], p[den])
    if {"spends", "leads"}.issubset(p):
        p["CPL"] = _safe_ratio(p.spends, p.leads)
    if {"spends", "enrollments"}.issubset(p):
        p["CAC"] = _safe_ratio(p.spends, p.enrollments)
    if {"gsv_cr", "enrollments"}.issubset(p):
        p["GSV_per_enrollment"] = _safe_ratio(p.gsv_cr, p.enrollments)

    cat_rows = []
    # Category is the analytical level; tech_management is the parent Vertical.
    # The page filters to one vertical before this function is called.
    for cat, g in p.groupby("category", dropna=False):
        row = {
            "category": str(cat), "tech_management": str(g.tech_management.iloc[0]),
            "programs": g.program_code.nunique(),
            "leads": g.leads.sum(min_count=1), "warm_leads": g.warm_leads.sum(min_count=1) if "warm_leads" in g else np.nan,
            "applications": g.applications.sum(min_count=1), "enrollments": g.enrollments.sum(min_count=1),
            "gsv_cr": g.gsv_cr.sum(min_count=1), "spends": g.spends.sum(min_count=1) if "spends" in g else np.nan,
        }
        row["L2W"] = row["warm_leads"] / row["leads"] if pd.notna(row["warm_leads"]) and pd.notna(row["leads"]) and row["leads"] > 0 else np.nan
        row["L2A"] = row["applications"] / row["leads"] if pd.notna(row["applications"]) and pd.notna(row["leads"]) and row["leads"] > 0 else np.nan
        row["W2A"] = row["applications"] / row["warm_leads"] if pd.notna(row["applications"]) and pd.notna(row["warm_leads"]) and row["warm_leads"] > 0 else np.nan
        row["A2E"] = row["enrollments"] / row["applications"] if pd.notna(row["enrollments"]) and pd.notna(row["applications"]) and row["applications"] > 0 else np.nan
        row["L2E"] = row["enrollments"] / row["leads"] if pd.notna(row["enrollments"]) and pd.notna(row["leads"]) and row["leads"] > 0 else np.nan
        row["CPL"] = row["spends"] / row["leads"] if pd.notna(row["spends"]) and pd.notna(row["leads"]) and row["leads"] > 0 else np.nan
        row["CAC"] = row["spends"] / row["enrollments"] if pd.notna(row["spends"]) and pd.notna(row["enrollments"]) and row["enrollments"] > 0 else np.nan
        row["GSV_per_enrollment"] = row["gsv_cr"] / row["enrollments"] if pd.notna(row["gsv_cr"]) and pd.notna(row["enrollments"]) and row["enrollments"] > 0 else np.nan
        for actual, plan, label in [
            ("leads", "Planned_Lead", "Lead_plan_attainment"),
            ("applications", "Planned_Application", "Application_plan_attainment"),
            ("enrollments", "Planned_Enrollment", "Enrollment_plan_attainment"),
            ("gsv_cr", "Planned_GSV", "GSV_plan_attainment"),
            ("spends", "Planned_Budget", "Budget_utilization"),
        ]:
            if plan in g:
                pv = g[plan].sum(min_count=1)
                row[label] = row[actual] / pv if pd.notna(row[actual]) and pd.notna(pv) and pv > 0 else np.nan
                row[label.replace("_attainment", "_plan").replace("_utilization", "_plan")] = pv
        cat_rows.append(row)
    categories = pd.DataFrame(cat_rows)
    if not categories.empty:
        categories["conversion_sample_status"] = np.where(
            (categories["programs"] >= 8) & (categories["leads"] >= 1000),
            "Sufficient for category conversion ranking",
            "Small sample - interpret conversion cautiously",
        )

    # Each program_code is the individual program batch identifier.
    batch_cols = [c for c in [
        "program_code", "program_name", "tech_management", "category", "financial_year",
        "program_batch_code", "batch_number", "New_Repeat", "institute_name_new",
        "program_fee_bucket", "leads", "warm_leads", "applications", "enrollments",
        "gsv_cr", "spends", "Planned_Lead", "Planned_Application", "Planned_Enrollment",
        "Planned_GSV", "Planned_Budget", "L2W", "L2A", "W2A", "A2E", "L2E",
        "CPL", "CAC", "GSV_per_enrollment",
    ] if c in p]
    batches = p[batch_cols].copy()
    if "leads" not in batches:
        batches["leads"] = np.nan

    # Category-relative batch performance: require enough lead volume to reduce noise.
    batches["category_l2e"] = batches.category.map(categories.set_index("category")["L2E"])
    batches["l2e_gap_pp"] = (batches.L2E - batches.category_l2e) * 100
    batches["batch_status"] = "Near category average"
    for cat, idx in batches.groupby("category").groups.items():
        g = batches.loc[idx]
        eligible = g[g.leads >= 30]
        if len(eligible) < 4:
            continue
        q25 = eligible.L2E.quantile(.25)
        q75 = eligible.L2E.quantile(.75)
        batches.loc[idx, "batch_status"] = "Insufficient lead volume"
        batches.loc[eligible.index, "batch_status"] = "Near category average"
        batches.loc[eligible[eligible.L2E <= q25].index, "batch_status"] = "Underperforming"
        batches.loc[eligible[eligible.L2E >= q75].index, "batch_status"] = "Top quartile"

    # Explain stage-level gap against the category's top quartile and estimate scenarios.
    diagnostics = []
    for _, b in batches.iterrows():
        cat = b.category
        if b.get("batch_status") != "Underperforming":
            continue
        peer = batches[(batches.category == cat) & (batches.leads >= 30)]
        if len(peer) < 4 or pd.isna(b.leads) or b.leads < 30:
            continue
        stage_q = {m: peer[m].quantile(.75) for m in ["L2W", "L2A", "A2E", "L2E"] if m in peer}
        own = {m: b.get(m, np.nan) for m in stage_q}
        gaps = {m: max(0.0, float(stage_q[m] - own[m])) for m in stage_q
                if pd.notna(stage_q[m]) and pd.notna(own[m])}
        # Diagnose largest stage gap in percentage points; L2E is an outcome, not added to stage opportunity.
        actionable = {m: v for m, v in gaps.items() if m in ("L2A", "A2E", "L2W")}
        driver = max(actionable, key=actionable.get) if actionable else None
        l2e_q = stage_q.get("L2E", np.nan)
        scenario = max(0.0, (l2e_q - b.L2E) * b.leads) if pd.notna(l2e_q) and pd.notna(b.L2E) else np.nan
        if driver == "L2A":
            action = "Review lead qualification, audience targeting, campaign/source mix and application-start journey."
        elif driver == "A2E":
            action = "Review application follow-up speed, counselling, eligibility friction, fee objections and payment completion."
        elif driver == "L2W":
            action = "Review lead validity, targeting, contactability and speed-to-lead."
        else:
            action = "Review the full funnel and compare against the strongest matched category peers."
        diagnostics.append({
            "program_code": b.program_code, "program_name": b.get("program_name"),
            "tech_management": b.tech_management, "category": cat, "leads": b.leads, "enrollments": b.get("enrollments"),
            "batch_L2E": b.get("L2E"), "category_L2E": b.get("category_l2e"),
            "top_quartile_L2E": l2e_q, "L2E_gap_pp_to_top_quartile": (l2e_q-b.L2E)*100 if pd.notna(l2e_q) and pd.notna(b.L2E) else np.nan,
            "estimated_enrollment_scenario": scenario, "largest_funnel_gap": driver,
            "batch_stage_rate": own.get(driver, np.nan) if driver else np.nan,
            "top_quartile_stage_rate": stage_q.get(driver, np.nan) if driver else np.nan,
            "recommended_investigation": action,
        })
    diagnostics = pd.DataFrame(diagnostics)
    return {"programs": p, "categories": categories, "batches": batches, "diagnostics": diagnostics}


def render_category_analysis(st, raw_df: pd.DataFrame) -> None:
    """Compact category report using the original numbered analysis structure."""
    st.markdown("## Category Analysis")
    st.caption("Analyse category performance within selected Verticals, with optional program-level drill-down.")

    programs, meta = prepare_programs(raw_df)
    if programs.empty:
        if meta.get("missing_columns"):
            st.error("Category data is missing required columns: " + ", ".join(meta["missing_columns"]))
        else:
            st.warning("No category data is available.")
        return

    def options(col, frame=None):
        frame = programs if frame is None else frame
        if col not in frame.columns:
            return []
        return sorted(frame[col].dropna().astype(str).unique().tolist())

    # Preserve the approved filter experience: Vertical > Category, with FY and optional programs.
    with st.expander("Choose your analysis scope", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            verticals = options("tech_management")
            selected_verticals = st.multiselect(
                "Vertical", verticals, default=verticals, key="cat_v5_verticals"
            )
        vertical_scope = programs[
            programs.tech_management.astype(str).isin(selected_verticals)
        ].copy() if selected_verticals else programs.iloc[0:0].copy()
        with c2:
            categories = options("category", vertical_scope)
            selected_categories = st.multiselect(
                "Category", categories, default=[],
                key="cat_v5_categories",
                help="Category values are taken from the CSV category column and limited to selected Verticals.",
                disabled=not selected_verticals,
            )
        with c3:
            fys = options("financial_year", vertical_scope)
            selected_fys = st.multiselect(
                "Financial year", fys, default=fys, key="cat_v5_fy"
            ) if fys else []

        filtered = vertical_scope[
            vertical_scope.category.astype(str).isin(selected_categories)
        ].copy() if selected_categories else vertical_scope.iloc[0:0].copy()
        if "financial_year" in filtered:
            filtered = filtered[filtered.financial_year.astype(str).isin(selected_fys)]

        available_codes = filtered.program_code.astype(str).drop_duplicates().tolist()
        name_rows = filtered.drop_duplicates("program_code")
        name_map = dict(zip(
            name_rows.program_code.astype(str),
            name_rows.program_name.fillna("Program").astype(str)
        )) if "program_name" in name_rows else {}
        with st.expander("Narrow to specific program batches (optional)", expanded=False):
            selected_codes = st.multiselect(
                "Programs / batches",
                options=available_codes,
                default=[],
                key="cat_v5_programs",
                format_func=lambda code: f"{name_map.get(code, 'Program')} | {code}",
                help="Leave empty to include all program batches under the selected Vertical and Category filters.",
                disabled=not selected_categories,
            )

    if not selected_verticals or not selected_categories:
        st.info("Select at least one Vertical and one Category to generate the analysis.")
        return
    if filtered.empty:
        st.warning("No program batches match the selected filters.")
        return

    scope = filtered[
        filtered.program_code.astype(str).isin(selected_codes)
    ].copy() if selected_codes else filtered.copy()
    if scope.empty:
        st.warning("No program batches remain in this selection.")
        return

    built = _build_management_tables(scope)
    cat_table, batch_table = built["categories"], built["batches"]
    if cat_table.empty or batch_table.empty:
        st.info("There is not enough program-level data for this selection.")
        return

    def n(v, d=0):
        return "N/A" if pd.isna(v) else f"{v:,.{d}f}"
    def p(v):
        return "N/A" if pd.isna(v) else f"{v:.2%}"

    st.markdown(
        f"**Analysis scope:** {len(selected_verticals)} Vertical(s) · "
        f"{len(selected_categories)} Category(s) · {scope.program_code.nunique():,} program batches"
    )

    # 1. Category comparison
    st.markdown("### 1. Category comparison")
    if len(cat_table) > 1:
        best_volume = cat_table.sort_values("enrollments", ascending=False).iloc[0]
        valid = cat_table.dropna(subset=["L2E"])
        best_conversion = valid.sort_values("L2E", ascending=False).iloc[0] if not valid.empty else None
        msg = (
            f"**{best_volume.category}** contributes the highest enrollment volume "
            f"({n(best_volume.enrollments)} enrollments from {n(best_volume.leads)} leads)."
        )
        if best_conversion is not None:
            msg += (
                f" **{best_conversion.category}** has the strongest lead-to-enrollment conversion "
                f"at {p(best_conversion.L2E)}."
            )
        st.write(msg)
    else:
        c = cat_table.iloc[0]
        st.write(
            f"**{c.category}** generated {n(c.leads)} leads and {n(c.enrollments)} enrollments "
            f"with lead-to-enrollment conversion of {p(c.L2E)}."
        )
    comparison_cols = [x for x in [
        "category", "programs", "leads", "warm_leads", "applications",
        "enrollments", "L2W", "L2A", "A2E", "L2E", "gsv_cr"
    ] if x in cat_table.columns]
    comparison = cat_table[comparison_cols].copy()
    for x in ["L2W", "L2A", "A2E", "L2E"]:
        if x in comparison:
            comparison[x] = comparison[x].map(p)
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    # Interactive category contribution pie chart.
    # A self-contained SVG avoids optional plotting packages and supports real slice labels.
    metric_options = [
        ("Leads", "leads"),
        ("Warm Leads", "warm_leads"),
        ("Applications", "applications"),
        ("Enrollments", "enrollments"),
        ("GSV", "gsv_cr"),
        ("Spend", "spends"),
        ("Paid Leads", "paid_leads"),
        ("Paid Enrollments", "paid_enrollments"),
        ("Paid GSV", "paid_gsv_cr"),
        ("Paid Spend", "paid_spends"),
    ]
    available_metrics = [(label, col) for label, col in metric_options if col in cat_table.columns]
    if available_metrics:
        chart_title_col, chart_metric_col = st.columns([4, 1])
        with chart_title_col:
            st.markdown("**Selected category contribution**")
            st.caption("3D-style category shares, with category names and percentages outside the slices.")
        with chart_metric_col:
            selected_metric_label = st.selectbox(
                "Metric",
                [label for label, _ in available_metrics],
                index=next((i for i, (label, _) in enumerate(available_metrics) if label == "Enrollments"), 0),
                key="cat_contribution_metric",
                label_visibility="collapsed",
            )
        selected_metric_col = dict(available_metrics)[selected_metric_label]
        pie_data = cat_table[["category", selected_metric_col]].copy()
        pie_data[selected_metric_col] = pd.to_numeric(pie_data[selected_metric_col], errors="coerce")
        pie_data = pie_data.dropna(subset=[selected_metric_col])
        pie_data = pie_data[pie_data[selected_metric_col] > 0].copy()
        pie_data = pie_data.rename(columns={selected_metric_col: "value"})
        total_metric_value = pie_data["value"].sum()

        if len(pie_data) > 0 and total_metric_value > 0:
            pie_data["share"] = pie_data["value"] / total_metric_value

            # SVG canvas: transparent background; elliptical top surface gives perspective.
            width, height = 1120, 500
            cx, cy = 430, 220
            rx, ry = 185, 145
            depth = 28
            palette = [
                "#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1",
                "#76B7B2", "#EDC948", "#FF9DA7", "#9C755F", "#BAB0AC",
                "#86BCB6", "#D37295", "#8CD17D", "#B6992D", "#499894",
                "#D4A6C8", "#79706E", "#FABFD2", "#A0CBE8", "#FFBE7D",
            ]

            def polar_point(angle, y_offset=0):
                return (
                    cx + rx * math.cos(angle),
                    cy + y_offset + ry * math.sin(angle),
                )

            def wedge_path(a0, a1, y_offset=0):
                x0, y0 = polar_point(a0, y_offset)
                x1, y1 = polar_point(a1, y_offset)
                large = 1 if (a1 - a0) > math.pi else 0
                return (
                    f"M {cx:.2f} {cy + y_offset:.2f} "
                    f"L {x0:.2f} {y0:.2f} "
                    f"A {rx} {ry} 0 {large} 1 {x1:.2f} {y1:.2f} Z"
                )

            def shade(hex_color, factor=0.55):
                h = hex_color.lstrip("#")
                rgb = [int(h[i:i+2], 16) for i in (0, 2, 4)]
                return "#" + "".join(f"{int(v*factor):02x}" for v in rgb)

            svg_parts = [
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
                'width="100%" role="img" aria-label="3D-style category contribution pie chart">',
                '<style>text{font-family:Arial,sans-serif}.cat-label{fill:#ffffff;font-size:15px;font-weight:600}.cat-value{fill:#dceaff;font-size:14px;font-weight:500}</style>',
            ]

            # Build slice geometry and draw the extruded lower faces first.
            angle = -math.pi / 2
            slices = []
            rows = pie_data.reset_index(drop=True)
            for i, row in rows.iterrows():
                share = float(row["share"])
                next_angle = angle + 2 * math.pi * share
                color = palette[i % len(palette)]
                slices.append((row, angle, next_angle, color))
                angle = next_angle

            # Lower offset layer; the top surface covers its upper half, leaving a visible side wall.
            for row, a0, a1, color in slices:
                svg_parts.append(
                    f'<path d="{wedge_path(a0, a1, depth)}" fill="{shade(color)}" '
                    'stroke="#17264a" stroke-width="1"/>'
                )

            # Top pie surface.
            for row, a0, a1, color in slices:
                svg_parts.append(
                    f'<path d="{wedge_path(a0, a1, 0)}" fill="{color}" '
                    'stroke="#ffffff" stroke-width="1.6"/>'
                )

            # Outside labels with leader lines; alternate left/right based on slice midpoint.
            for i, (row, a0, a1, color) in enumerate(slices):
                mid = (a0 + a1) / 2
                x1, y1 = polar_point(mid, 0)
                x2, y2 = polar_point(mid, 24)
                right_side = math.cos(mid) >= 0
                x3 = x2 + (68 if right_side else -68)
                y3 = y2 + (i % 3 - 1) * 9
                anchor = "start" if right_side else "end"
                tx = x3 + (7 if right_side else -7)
                category_text = html.escape(str(row["category"]))
                share_text = f'{float(row["share"]):.1%}'
                svg_parts.append(
                    f'<polyline points="{x1:.1f},{y1:.1f} {x2:.1f},{y2:.1f} {x3:.1f},{y3:.1f}" '
                    f'fill="none" stroke="{color}" stroke-width="1.8"/>'
                )
                svg_parts.append(
                    f'<text class="cat-label" x="{tx:.1f}" y="{y3-3:.1f}" text-anchor="{anchor}">'
                    f'{category_text}</text>'
                )
                svg_parts.append(
                    f'<text class="cat-value" x="{tx:.1f}" y="{y3+17:.1f}" text-anchor="{anchor}">'
                    f'{share_text}</text>'
                )

            svg_parts.append("</svg>")
            st.markdown("".join(svg_parts), unsafe_allow_html=True)
            st.caption(
                f"Showing {selected_metric_label.lower()} contribution across "
                f"{len(pie_data)} selected categories. Total: {total_metric_value:,.2f}."
            )
        else:
            st.info(f"No positive {selected_metric_label.lower()} values are available for the selected categories.")
    else:
        st.info("No supported numeric contribution metrics are available for the selected categories.")


    # 2. Category performance diagnosis
    st.markdown("### 2. Category performance diagnosis")
    for _, c in cat_table.sort_values("enrollments", ascending=False).iterrows():
        batches = batch_table[batch_table.category.astype(str) == str(c.category)]
        eligible = batches[batches.leads >= 30]
        st.markdown(f"**{c.category}**")
        paragraph = (
            f"This category has **{int(c.programs):,} program batches**, generating "
            f"**{n(c.leads)} leads**, **{n(c.applications)} applications** and "
            f"**{n(c.enrollments)} enrollments** (L2E: **{p(c.L2E)}**)."
        )
        if len(eligible) >= 2:
            top = eligible.sort_values("L2E", ascending=False).iloc[0]
            low = eligible.sort_values("L2E", ascending=True).iloc[0]
            paragraph += (
                f" The strongest eligible batch is **{top.program_code}** ({p(top.L2E)} L2E), "
                f"while **{low.program_code}** is lowest ({p(low.L2E)})."
            )
            if pd.notna(top.L2E) and pd.notna(low.L2E):
                paragraph += f" The conversion gap is {(top.L2E-low.L2E)*100:.2f} percentage points."
        else:
            paragraph += " There are too few batches with at least 30 leads for a reliable within-category ranking."
        st.write(paragraph)

    # 3. Funnel performance
    st.markdown("### 3. Funnel performance")
    total_leads = scope.leads.sum(min_count=1)
    total_warm = scope.warm_leads.sum(min_count=1)
    total_apps = scope.applications.sum(min_count=1)
    total_enrol = scope.enrollments.sum(min_count=1)
    funnel = pd.DataFrame([
        {"Funnel stage": "Lead → Warm lead", "Volume": total_warm, "Conversion": total_warm / total_leads if pd.notna(total_leads) and total_leads > 0 else np.nan},
        {"Funnel stage": "Lead → Application", "Volume": total_apps, "Conversion": total_apps / total_leads if pd.notna(total_leads) and total_leads > 0 else np.nan},
        {"Funnel stage": "Application → Enrollment", "Volume": total_enrol, "Conversion": total_enrol / total_apps if pd.notna(total_apps) and total_apps > 0 else np.nan},
        {"Funnel stage": "Lead → Enrollment", "Volume": total_enrol, "Conversion": total_enrol / total_leads if pd.notna(total_leads) and total_leads > 0 else np.nan},
    ])
    funnel["Conversion"] = funnel["Conversion"].map(p)
    st.dataframe(funnel, use_container_width=True, hide_index=True)

    # 4. Program-level year-over-year comparison (consolidated across batch codes).
    st.markdown("### 4. Program performance — year-over-year comparison")
    st.caption(
        "Batch-code suffixes are removed from program names and all corresponding batches "
        "are consolidated into one program within each financial year."
    )

    # The selector mirrors the category contribution chart and allows the user
    # to switch the comparison metric without changing the selected scope.
    yoy_metric_options = [
        ("Leads", "leads"),
        ("Warm Leads", "warm_leads"),
        ("Applications", "applications"),
        ("Enrollments", "enrollments"),
        ("Lead → Warm Lead Conversion", "L2W"),
        ("Lead → Application Conversion", "L2A"),
        ("Warm Lead → Application Conversion", "W2A"),
        ("Application → Enrollment Conversion", "A2E"),
        ("Lead → Enrollment Conversion", "L2E"),
        ("GSV", "gsv_cr"),
        ("Spend", "spends"),
        ("Cost per Lead", "CPL"),
        ("Cost per Acquisition", "CAC"),
        ("GSV per Enrollment", "GSV_per_enrollment"),
    ]
    yoy_available = [
        (label, col) for label, col in yoy_metric_options
        if col in batch_table.columns or col in {"CPL", "CAC", "GSV_per_enrollment"}
    ]
    yoy_metric_col1, yoy_metric_col2 = st.columns([3, 1])
    with yoy_metric_col1:
        st.write("Select a metric to compare across financial years.")
    with yoy_metric_col2:
        yoy_metric_label = st.selectbox(
            "Comparison metric",
            [label for label, _ in yoy_available],
            index=next(
                (i for i, (label, _) in enumerate(yoy_available) if label == "Enrollments"),
                0,
            ),
            key="cat_program_yoy_metric",
            label_visibility="collapsed",
        )
    yoy_metric = dict(yoy_available)[yoy_metric_label]

    # Derive a stable program identity from program_name, not program_code.
    # Names such as "..._01", "..._02", etc. are the same program with different batches.
    program_yoy_source = scope.copy()
    if "program_name" in program_yoy_source:
        program_yoy_source["_program_identity"] = (
            program_yoy_source["program_name"]
            .fillna(program_yoy_source["program_code"].astype(str))
            .astype(str).str.strip()
            .str.replace(r"[_-]\d+$", "", regex=True)
        )
    else:
        program_yoy_source["_program_identity"] = (
            program_yoy_source["program_code"].astype(str)
            .str.replace(r"[_-]\d+$", "", regex=True)
        )
    program_yoy_source["_program_display"] = (
        program_yoy_source["_program_identity"].str.replace("_", " ", regex=False)
    )

    # Aggregate all batch codes belonging to the same program in each FY.
    yoy_group_cols = ["_program_identity", "_program_display", "category", "financial_year"]
    additive_yoy_cols = [
        c for c in ["leads", "warm_leads", "applications", "enrollments", "gsv_cr", "spends"]
        if c in program_yoy_source.columns
    ]
    yoy_agg = {c: (lambda x: x.sum(min_count=1)) for c in additive_yoy_cols}
    program_yoy = (
        program_yoy_source.groupby(yoy_group_cols, dropna=False, as_index=False)
        .agg(yoy_agg)
    )
    for numerator, denominator, rate_name in [
        ("warm_leads", "leads", "L2W"),
        ("applications", "leads", "L2A"),
        ("applications", "warm_leads", "W2A"),
        ("enrollments", "applications", "A2E"),
        ("enrollments", "leads", "L2E"),
    ]:
        if numerator in program_yoy and denominator in program_yoy:
            program_yoy[rate_name] = _safe_ratio(
                program_yoy[numerator], program_yoy[denominator]
            )
    if {"spends", "leads"}.issubset(program_yoy.columns):
        program_yoy["CPL"] = _safe_ratio(program_yoy["spends"], program_yoy["leads"])
    if {"spends", "enrollments"}.issubset(program_yoy.columns):
        program_yoy["CAC"] = _safe_ratio(program_yoy["spends"], program_yoy["enrollments"])
    if {"gsv_cr", "enrollments"}.issubset(program_yoy.columns):
        program_yoy["GSV_per_enrollment"] = _safe_ratio(
            program_yoy["gsv_cr"], program_yoy["enrollments"]
        )

    # Sort fiscal years chronologically; display newest year first.
    available_fys = program_yoy["financial_year"].dropna().astype(str).unique().tolist()
    def _fy_sort_key(value):
        match = re.search(r"(\d+)", str(value))
        return int(match.group(1)) if match else -1
    fy_order = sorted(available_fys, key=_fy_sort_key)
    fy_desc = list(reversed(fy_order))

    if not program_yoy.empty and yoy_metric in program_yoy.columns:
        # Build one row per distinct program and retain the category as context.
        output_rows = []
        for (identity, display, category_name), group in program_yoy.groupby(
            ["_program_identity", "_program_display", "category"], dropna=False
        ):
            fy_values = {
                str(row["financial_year"]): row.get(yoy_metric, np.nan)
                for _, row in group.iterrows()
            }
            row_out = {
                "Program": display,
                "Category": category_name,
            }
            for fy in fy_desc:
                value = fy_values.get(fy, np.nan)
                if yoy_metric in {"L2W", "L2A", "W2A", "A2E", "L2E"}:
                    row_out[fy] = "N/A" if pd.isna(value) else f"{value:.2%}"
                elif yoy_metric in {"CPL", "CAC"}:
                    row_out[fy] = "N/A" if pd.isna(value) else f"₹{value:,.0f}"
                elif yoy_metric == "GSV_per_enrollment":
                    row_out[fy] = "N/A" if pd.isna(value) else f"{value:,.2f}"
                else:
                    row_out[fy] = "N/A" if pd.isna(value) else f"{value:,.0f}"

            # Year-over-year deltas compare each year with the immediately preceding FY.
            for idx in range(len(fy_desc) - 1):
                newer, older = fy_desc[idx], fy_desc[idx + 1]
                newer_value = fy_values.get(newer, np.nan)
                older_value = fy_values.get(older, np.nan)
                delta_key = f"{newer} vs {older}"
                if pd.isna(newer_value) or pd.isna(older_value) or older_value == 0:
                    row_out[delta_key] = "N/A"
                else:
                    pct_change = (newer_value - older_value) / abs(older_value) * 100
                    pp_change = (newer_value - older_value) * 100
                    # A decrease is favourable for acquisition-cost metrics.
                    favourable = pct_change <= 0 if yoy_metric in {"CPL", "CAC"} else pct_change >= 0
                    arrow = "🟢 ↑" if favourable else "🔴 ↓"
                    if yoy_metric in {"L2W", "L2A", "W2A", "A2E", "L2E"}:
                        row_out[delta_key] = f"{arrow} {pct_change:+.1f}% ({pp_change:+.2f} pp)"
                    else:
                        row_out[delta_key] = f"{arrow} {pct_change:+.1f}%"
            output_rows.append(row_out)

        yoy_display = pd.DataFrame(output_rows)
        # Sort by the latest available year for the selected metric.
        latest_fy = fy_desc[0] if fy_desc else None
        if latest_fy and latest_fy in yoy_display.columns:
            sort_values = program_yoy[program_yoy.financial_year.astype(str) == latest_fy][
                ["_program_display", "category", yoy_metric]
            ].copy()
            sort_values = sort_values.rename(columns={
                "_program_display": "Program", yoy_metric: "_latest_metric"
            })
            yoy_display = yoy_display.merge(
                sort_values, on=["Program", "Category"], how="left"
            ).sort_values("_latest_metric", ascending=False, na_position="last")
            yoy_display = yoy_display.drop(columns=["_latest_metric"])

        st.write(
            f"Comparing **{yoy_display['Program'].nunique():,} distinct programs** "
            f"across **{len(fy_desc)} financial years** using **{yoy_metric_label}**. "
            "Repeated batch codes are consolidated under the same program name."
        )
        st.dataframe(yoy_display, use_container_width=True, hide_index=True)

        # Compact interpretation based on the latest FY and the preceding FY.
        if len(fy_desc) >= 2:
            latest_fy, previous_fy = fy_desc[0], fy_desc[1]
            latest_values = program_yoy[program_yoy.financial_year.astype(str) == latest_fy]
            previous_values = program_yoy[program_yoy.financial_year.astype(str) == previous_fy]
            latest_valid = latest_values.dropna(subset=[yoy_metric]) if yoy_metric in latest_values else pd.DataFrame()
            if not latest_valid.empty:
                best_row = latest_valid.sort_values(yoy_metric, ascending=(yoy_metric in {"CPL", "CAC"})).iloc[0]
                st.write(
                    f"**Latest-year leader ({latest_fy}):** {best_row['_program_display']} "
                    f"({best_row[yoy_metric]:,.2f}{'%' if yoy_metric in {'L2W','L2A','W2A','A2E','L2E'} else ''})."
                )
            comparable = latest_values.merge(
                previous_values,
                on=["_program_identity", "_program_display", "category"],
                suffixes=("_new", "_old"),
            )
            metric_new, metric_old = f"{yoy_metric}_new", f"{yoy_metric}_old"
            if metric_new in comparable and metric_old in comparable:
                comparable = comparable.dropna(subset=[metric_new, metric_old])
                comparable = comparable[comparable[metric_old] != 0]
                if not comparable.empty:
                    comparable["_change"] = (
                        (comparable[metric_new] - comparable[metric_old])
                        / comparable[metric_old].abs()
                    )
                    improved = (
                        comparable["_change"] < 0
                        if yoy_metric in {"CPL", "CAC"}
                        else comparable["_change"] > 0
                    )
                    st.caption(
                        f"{int(improved.sum())} of {len(comparable)} comparable programs improved "
                        f"from {previous_fy} to {latest_fy} on {yoy_metric_label}. "
                        "For CPL/CAC, a decrease is treated as improvement; for other metrics, an increase is treated as improvement."
                    )
    else:
        st.info("No data is available for the selected metric and financial-year scope.")

    # 5. Good vs. weak program comparison — same-category peer benchmark.
    st.markdown("### 5. What good-performing programs do differently")
    st.caption(
        "Within each selected category, top- and bottom-quartile batches are compared using "
        "the same lead-volume eligibility threshold. Green arrows indicate a stronger rate; "
        "red arrows indicate a weaker rate."
    )

    def direction(value, benchmark, higher_is_better=True):
        if pd.isna(value) or pd.isna(benchmark):
            return "— N/A"
        positive = value >= benchmark if higher_is_better else value <= benchmark
        arrow = "🟢 ↑" if positive else "🔴 ↓"
        return f"{arrow} {p(value)}"

    def rate_gap(value, benchmark):
        if pd.isna(value) or pd.isna(benchmark):
            return "N/A"
        diff = (value - benchmark) * 100
        arrow = "🟢 ↑" if diff >= 0 else "🔴 ↓"
        return f"{arrow} {diff:+.2f} pp"

    comparisons = []
    action_rows = []
    # Use the full filtered category pool for fair peer benchmarks, even when the
    # user has narrowed the displayed program list to only a few batches.
    benchmark_batches = _build_management_tables(filtered)["batches"]
    for category_name, selected_group in batch_table.groupby("category", dropna=False):
        peer_group = benchmark_batches[
            benchmark_batches.category.astype(str) == str(category_name)
        ]
        eligible = peer_group[
            (pd.to_numeric(peer_group["leads"], errors="coerce") >= 30)
            & peer_group["L2E"].notna()
        ].copy()
        selected_eligible = selected_group[
            (pd.to_numeric(selected_group["leads"], errors="coerce") >= 30)
            & selected_group["L2E"].notna()
        ].copy()
        if len(eligible) < 4:
            continue
        q25, q75 = eligible["L2E"].quantile([0.25, 0.75])
        weak = selected_eligible[selected_eligible["L2E"] <= q25].copy()
        good = eligible[eligible["L2E"] >= q75].copy()
        if weak.empty:
            continue
        if weak.empty or good.empty:
            continue

        rate_cols = ["L2W", "L2A", "W2A", "A2E", "L2E"]
        good_median = {m: good[m].median() for m in rate_cols if m in good}
        weak_median = {m: weak[m].median() for m in rate_cols if m in weak}
        comparisons.append({
            "Category": category_name,
            "Good batches": len(good),
            "Weak batches": len(weak),
            "Good median L2E": good_median.get("L2E", np.nan),
            "Weak median L2E": weak_median.get("L2E", np.nan),
            "L2E gap (pp)": (
                (good_median.get("L2E", np.nan) - weak_median.get("L2E", np.nan)) * 100
                if pd.notna(good_median.get("L2E", np.nan)) and pd.notna(weak_median.get("L2E", np.nan))
                else np.nan
            ),
        })

        for _, weak_batch in weak.sort_values("L2E").iterrows():
            # Identify the largest actionable stage gap against the good cohort median.
            candidates = []
            for metric, label in [
                ("L2W", "Lead → Warm lead"),
                ("L2A", "Lead → Application"),
                ("A2E", "Application → Enrollment"),
            ]:
                own_rate = weak_batch.get(metric, np.nan)
                good_rate = good_median.get(metric, np.nan)
                if pd.notna(own_rate) and pd.notna(good_rate):
                    candidates.append((good_rate - own_rate, metric, label, own_rate, good_rate))
            candidates = [x for x in candidates if x[0] > 0]
            if not candidates:
                continue
            gap, metric, label, own_rate, good_rate = max(candidates, key=lambda x: x[0])

            # Illustrative volume opportunity, holding downstream conversion constant.
            if metric == "L2W":
                downstream = weak_batch.get("W2A", np.nan) * weak_batch.get("A2E", np.nan)
                opportunity = weak_batch.leads * gap * downstream if pd.notna(downstream) else np.nan
            elif metric == "L2A":
                downstream = weak_batch.get("A2E", np.nan)
                opportunity = weak_batch.leads * gap * downstream if pd.notna(downstream) else np.nan
            else:
                opportunity = weak_batch.applications * gap if pd.notna(weak_batch.get("applications", np.nan)) else np.nan

            action_map = {
                "L2W": "Audit lead validity and source mix; tighten audience targeting, improve contactability, and shorten speed-to-lead.",
                "L2A": "Compare lead qualification and counselling-to-application practices; remove application friction and review source-level lead quality.",
                "A2E": "Review counselling follow-up, eligibility objections, fee/payment drop-offs, and application completion discipline.",
            }
            action_rows.append({
                "category": category_name,
                "program_code": str(weak_batch.program_code),
                "program_name": weak_batch.get("program_name", "Program"),
                "leads": weak_batch.leads,
                "enrollments": weak_batch.enrollments,
                "L2E": weak_batch.L2E,
                "metric": metric,
                "stage_label": label,
                "own_rate": own_rate,
                "good_rate": good_rate,
                "gap_pp": gap * 100,
                "opportunity": opportunity,
                "action": action_map[metric],
                "good": good,
            })

    if comparisons:
        comp_df = pd.DataFrame(comparisons)
        comp_display = comp_df.copy()
        for col in ["Good median L2E", "Weak median L2E"]:
            comp_display[col] = comp_display[col].map(p)
        comp_display["L2E gap (pp)"] = comp_display["L2E gap (pp)"].map(
            lambda x: "N/A" if pd.isna(x) else f"🟢 ↑ {x:.2f} pp"
        )
        st.dataframe(comp_display, use_container_width=True, hide_index=True)

        if action_rows:
            action_df = pd.DataFrame(action_rows).sort_values(
                "opportunity", ascending=False, na_position="last"
            )
            for _, item in action_df.head(8).iterrows():
                st.markdown(
                    f"**{item['category']} — {item['program_code']}** "
                    f"({item['program_name']})"
                )
                st.write(
                    f"🔴 ↓ **{item['stage_label']} conversion: {p(item['own_rate'])}** "
                    f"versus 🟢 ↑ **{p(item['good_rate'])}** median among top-performing "
                    f"batches in the same category — a gap of **{item['gap_pp']:.2f} percentage points**."
                )
                st.write(
                    f"**Where it went wrong:** this is the largest observed actionable funnel-stage "
                    f"gap for this batch versus the category's good-performing peer group. "
                    f"It generated {n(item['leads'])} leads and {n(item['enrollments'])} enrollments "
                    f"(L2E: {p(item['L2E'])})."
                )
                st.write(
                    f"**What to do:** {item['action']} "
                    f"**Illustrative opportunity:** approximately {n(item['opportunity'], 1)} "
                    f"additional enrollments if the identified stage rate reached the good-peer "
                    f"median and downstream conversion stayed unchanged."
                )

            # One compact proof table for the selected weak batches.
            proof = action_df.head(8)[[
                "category", "program_code", "leads", "enrollments",
                "L2E", "stage_label", "own_rate", "good_rate", "gap_pp", "opportunity"
            ]].copy()
            proof.columns = [
                "Category", "Weak program code", "Leads", "Enrollments",
                "Lead-to-enrollment", "Weakest stage", "Weak batch rate",
                "Good-peer median", "Gap (pp)", "Illustrative extra enrollments"
            ]
            for col in ["Lead-to-enrollment", "Weak batch rate", "Good-peer median"]:
                proof[col] = proof[col].map(p)
            proof["Gap (pp)"] = proof["Gap (pp)"].map(lambda x: f"🔴 ↓ {x:.2f}" if pd.notna(x) else "N/A")
            st.caption("Evidence for the program-level recommendations above.")
            st.dataframe(proof, use_container_width=True, hide_index=True)
        else:
            st.write(
                "No selected bottom-quartile batch showed a positive gap in the measured "
                "actionable funnel stages versus its category's top-quartile peers."
            )
    else:
        st.info(
            "A good-versus-weak comparison needs at least four program batches with 30 or more "
            "leads and a valid lead-to-enrollment rate in a category. The current selection "
            "does not meet that threshold."
        )

    st.caption(
        "Benchmark interpretation: top/bottom quartiles are descriptive peer groups, not proof "
        "of causation. Estimated extra enrollments are scenarios, not forecasts; they assume "
        "the selected stage improves to the good-peer median while downstream rates remain constant."
    )

    # 6. What to improve
    st.markdown("### 6. Additional improvement opportunities")
    diagnostics = built.get("diagnostics", pd.DataFrame())
    shown = 0
    if not diagnostics.empty:
        selected_diagnostics = diagnostics[
            diagnostics.program_code.astype(str).isin(batch_table.program_code.astype(str))
        ].sort_values("estimated_enrollment_scenario", ascending=False)
        for _, d in selected_diagnostics.head(5).iterrows():
            st.write(
                f"- **{d.program_code}** ({d.category}): largest identified funnel gap is "
                f"{d.largest_funnel_gap or 'not clear'}. Current stage conversion is "
                f"{p(d.batch_stage_rate)} versus {p(d.top_quartile_stage_rate)} for the "
                f"category's top-quartile peer benchmark. Estimated enrollment opportunity: "
                f"{n(d.estimated_enrollment_scenario)} if the historical benchmark were reached. "
                f"**Next step:** {d.recommended_investigation}"
            )
            shown += 1
    if not shown:
        st.write(
            "No reliable underperforming-batch opportunity was identified in this selection. "
            "Use the funnel comparison above to decide which stage warrants investigation."
        )

    # 7. Commercial outcomes and data caveats
    st.markdown("### 7. Commercial outcomes")
    if "gsv_cr" in batch_table and batch_table.gsv_cr.notna().any():
        gsv = batch_table.gsv_cr.sum(min_count=1)
        st.write(
            f"The selected program batches generated **{n(gsv, 2)} Cr GSV** across "
            f"**{n(total_enrol)} enrollments**."
        )
        commercial_cols = [x for x in ["category", "gsv_cr", "enrollments", "GSV_per_enrollment", "spends", "CAC"] if x in cat_table]
        if commercial_cols:
            st.dataframe(cat_table[commercial_cols], use_container_width=True, hide_index=True)
        if "spends" in scope:
            coverage = scope.spends.notna().mean()
            st.caption(f"Spend is available for {coverage:.1%} of selected program batches; interpret CPL/CAC only with this coverage limitation.")
    else:
        st.write("GSV is not available for the selected program batches.")

    result = {
        "categories": cat_table.to_dict(orient="records"),
        "batches": batch_table.to_dict(orient="records"),
        "diagnostics": diagnostics.to_dict(orient="records") if not diagnostics.empty else [],
        "scope": {
            "verticals": selected_verticals, "categories": selected_categories,
            "financial_years": selected_fys, "program_codes": scope.program_code.astype(str).tolist(),
        },
    }
    _render_category_chat(st, raw_df, scope, result)


def _category_chat_context(raw: pd.DataFrame, programs: pd.DataFrame, result: dict) -> str:
    """Build auditable context for category chat: source schema, analysis and records."""
    parts = [
        "CATEGORY INTELLIGENCE CHAT CONTEXT",
        f"Source rows loaded: {len(raw):,}",
        f"Unique programs after source-row consolidation: {programs.program_code.nunique():,}",
        "Primary grouping: Vertical (tech_management) then Category. Program-level metrics are summed across source_created rows; program attributes are retained once.",
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
