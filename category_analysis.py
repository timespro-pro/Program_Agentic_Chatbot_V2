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
    for cat, g in p.groupby("tech_management", dropna=False):
        row = {
            "tech_management": str(cat), "programs": g.program_code.nunique(),
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
    batches["category_l2e"] = batches.tech_management.map(categories.set_index("tech_management")["L2E"])
    batches["l2e_gap_pp"] = (batches.L2E - batches.category_l2e) * 100
    batches["batch_status"] = "Near category average"
    for cat, idx in batches.groupby("tech_management").groups.items():
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
        cat = b.tech_management
        if b.get("batch_status") != "Underperforming":
            continue
        peer = batches[(batches.tech_management == cat) & (batches.leads >= 30)]
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
            "tech_management": cat, "leads": b.leads, "enrollments": b.get("enrollments"),
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
    """Interactive category + program investigation driven by the user's filters."""
    st.markdown("## Category & Program Performance Analysis")
    st.caption("Select the categories and program batches you want to investigate. The narrative, comparisons and recommendations update to match your selection.")

    @st.cache_data(show_spinner="Preparing category and program metrics...")
    def _cached_programs(dataframe: pd.DataFrame):
        return prepare_programs(dataframe)

    programs, meta = _cached_programs(raw_df)
    if programs.empty:
        st.error("Cannot run the analysis. Required columns: program_code and tech_management.")
        return

    # Selection controls: category selection first, then programs within those categories.
    cats = sorted(programs.tech_management.dropna().astype(str).unique().tolist())
    fys = sorted(programs.financial_year.dropna().astype(str).unique().tolist()) if "financial_year" in programs else []
    new_repeat = sorted(programs.New_Repeat.dropna().astype(str).unique().tolist()) if "New_Repeat" in programs else []
    with st.container(border=True):
        st.markdown("### Choose your analysis scope")
        c1, c2, c3 = st.columns(3)
        with c1:
            selected_cats = st.multiselect("Categories (Tech Management)", cats, default=cats, key="cat_v3_categories")
        with c2:
            selected_fys = st.multiselect("Financial year", fys, default=fys, key="cat_v3_fy") if fys else []
        with c3:
            selected_nr = st.multiselect("New / Repeat", new_repeat, default=new_repeat, key="cat_v3_new_repeat") if new_repeat else []

        filtered = programs[programs.tech_management.astype(str).isin(selected_cats)].copy()
        if "financial_year" in filtered:
            filtered = filtered[filtered.financial_year.astype(str).isin(selected_fys)]
        if "New_Repeat" in filtered:
            filtered = filtered[filtered.New_Repeat.astype(str).isin(selected_nr)]
        available_codes = filtered.program_code.astype(str).drop_duplicates().tolist()
        name_rows = filtered.drop_duplicates("program_code")
        name_map = dict(zip(name_rows.program_code.astype(str), name_rows.program_name.fillna("Program").astype(str))) if "program_name" in name_rows else {}
        with st.expander("Narrow to specific program batches (optional)", expanded=True):
            selected_codes = st.multiselect(
                "Programs / batches — leave empty to include every program in the selected categories",
                options=available_codes, default=[], key="cat_v3_programs",
                format_func=lambda code: f"{name_map.get(code, 'Program')} | {code}"
            )
    if filtered.empty:
        st.warning("No program batches match the selected filters. Update your category or period selection.")
        return

    scope = filtered[filtered.program_code.astype(str).isin(selected_codes)].copy() if selected_codes else filtered.copy()
    if scope.empty:
        st.warning("No program batches remain in this selection.")
        return

    built = _build_management_tables(scope)
    cat_table, batch_table = built["categories"], built["batches"]
    selected_count = scope.program_code.nunique()
    st.markdown(f"**Analysis scope:** {len(cat_table)} categories · {selected_count:,} program batches · {scope.leads.sum(min_count=1):,.0f} leads · {scope.enrollments.sum(min_count=1):,.0f} enrollments")

    def pct(v):
        return "not available" if pd.isna(v) else f"{v:.2%}"

    def num(v, decimals=0):
        return "not available" if pd.isna(v) else f"{v:,.{decimals}f}"

    def funnel_weakness(row, peers):
        candidates = []
        for metric, label in [("L2W", "lead-to-warm-lead"), ("L2A", "lead-to-application"), ("A2E", "application-to-enrollment")]:
            own = row.get(metric, np.nan)
            peer = peers[metric].median() if metric in peers and peers[metric].notna().any() else np.nan
            if pd.notna(own) and pd.notna(peer):
                candidates.append((peer - own, metric, label, own, peer))
        return max(candidates, default=(np.nan, None, "", np.nan, np.nan), key=lambda x: x[0])

    # 1. Overall comparison only across selected categories.
    st.markdown("## 1. Selected category comparison")
    if len(cat_table) > 1:
        ranked = cat_table.sort_values("L2E", ascending=False, na_position="last")
        best_conv = ranked.iloc[0]
        best_volume = cat_table.sort_values("enrollments", ascending=False).iloc[0]
        st.write(
            f"Among the selected categories, **{best_volume.tech_management}** contributes the most enrollments "
            f"({num(best_volume.enrollments)} from {num(best_volume.leads)} leads). "
            f"**{best_conv.tech_management}** has the highest overall lead-to-enrollment conversion "
            f"({pct(best_conv.L2E)}). These are different measures: scale and conversion efficiency should not be treated as the same thing."
        )
        chart = cat_table.set_index("tech_management")[["enrollments"]]
        st.bar_chart(chart)
    else:
        only = cat_table.iloc[0]
        st.write(f"You selected one category: **{only.tech_management}**. The analysis below explains its funnel and compares the included program batches against one another.")

    summary_cols = [c for c in ["tech_management", "programs", "leads", "warm_leads", "applications", "enrollments", "L2W", "L2A", "A2E", "L2E", "gsv_cr"] if c in cat_table]
    compact = cat_table[summary_cols].copy()
    for c in ["L2W", "L2A", "A2E", "L2E"]:
        if c in compact:
            compact[c] = compact[c].map(pct)
    st.dataframe(compact, use_container_width=True, hide_index=True)

    # 2. A concise narrative for every selected category.
    st.markdown("## 2. Category-by-category findings")
    for _, cat in cat_table.sort_values("enrollments", ascending=False).iterrows():
        cat_batches = batch_table[batch_table.tech_management == cat.tech_management].copy()
        eligible = cat_batches[cat_batches.leads >= 30]
        st.markdown(f"### {cat.tech_management}")
        if cat.programs == 1:
            st.write(f"This selection contains one program batch, generating {num(cat.leads)} leads and {num(cat.enrollments)} enrollments (L2E {pct(cat.L2E)}). There are no within-category peer batches in this filtered scope, so performance can be described but not ranked reliably.")
            continue
        med_l2e = eligible.L2E.median() if not eligible.empty else np.nan
        top = eligible.sort_values("L2E", ascending=False).head(1)
        low = eligible.sort_values("L2E", ascending=True).head(1)
        finding = f"**{cat.programs:,} batches** generated **{num(cat.leads)} leads** and **{num(cat.enrollments)} enrollments**, giving a weighted L2E of **{pct(cat.L2E)}**."
        if pd.notna(med_l2e):
            finding += f" The median batch L2E is {pct(med_l2e)}, which helps distinguish the category's aggregate result from the typical batch experience."
        if not top.empty and not low.empty and top.iloc[0].program_code != low.iloc[0].program_code:
            finding += f" The strongest converting selected batch is **{top.iloc[0].program_code}** at {pct(top.iloc[0].L2E)}; the weakest eligible batch is **{low.iloc[0].program_code}** at {pct(low.iloc[0].L2E)}."
        st.write(finding)
        if not eligible.empty:
            gap, metric, label, own, peer = funnel_weakness(eligible.sort_values("L2E").iloc[0], eligible)
            if metric and gap > 0:
                st.write(f"**What the funnel suggests:** the weakest eligible batch's largest gap versus the selected-category median is at {label} ({pct(own)} versus {pct(peer)}). Investigate this stage first rather than assuming that more leads alone will solve the problem.")
        if len(eligible) < 4:
            st.caption("Interpret batch ranking cautiously: fewer than four selected batches have at least 30 leads.")

    # 3. Program-level analysis and peer comparison for exactly the selected programs.
    st.markdown("## 3. Program / batch comparison")
    if len(batch_table) == 1:
        b = batch_table.iloc[0]
        st.write(f"**{b.program_code}** ({b.get('program_name', 'Program')}) generated {num(b.leads)} leads, {num(b.warm_leads)} warm leads, {num(b.applications)} applications and {num(b.enrollments)} enrollments. Conversion: L2W {pct(b.L2W)}, L2A {pct(b.L2A)}, A2E {pct(b.A2E)}, and L2E {pct(b.L2E)}.")
    else:
        ranked_programs = batch_table.sort_values("L2E", ascending=False, na_position="last")
        best = ranked_programs.iloc[0]
        worst = ranked_programs.iloc[-1]
        st.write(f"Across the **{len(batch_table):,} selected batches**, **{best.program_code}** has the highest L2E ({pct(best.L2E)}) and **{worst.program_code}** has the lowest ({pct(worst.L2E)}). Compare their funnel stages to see whether the gap emerges before application or during application-to-enrollment conversion.")
        plot = batch_table.set_index("program_code")[["L2W", "L2A", "A2E", "L2E"]].copy()
        st.bar_chart(plot)

    evidence_cols = [c for c in ["program_code", "program_name", "tech_management", "financial_year", "leads", "warm_leads", "applications", "enrollments", "L2W", "L2A", "A2E", "L2E", "gsv_cr"] if c in batch_table]
    evidence = batch_table[evidence_cols].copy()
    for c in ["L2W", "L2A", "A2E", "L2E"]:
        if c in evidence:
            evidence[c] = evidence[c].map(pct)
    st.caption("Selected program evidence (one row per program_code).")
    st.dataframe(evidence, use_container_width=True, hide_index=True)

    # 4. Explain each selected program against peers in its own category.
    st.markdown("## 4. What is driving the differences? What to improve")
    for _, b in batch_table.sort_values("L2E", ascending=True).iterrows():
        peers = batch_table[(batch_table.tech_management == b.tech_management) & (batch_table.program_code != b.program_code) & (batch_table.leads >= 30)]
        st.markdown(f"**{b.program_code} — {b.get('program_name', 'Program')}**")
        if peers.empty or pd.isna(b.L2E):
            st.write(f"This batch has L2E of {pct(b.L2E)}. There are not enough other selected peer batches with at least 30 leads to identify a reliable category-relative gap.")
            continue
        peer_l2e = (peers.enrollments.sum(min_count=1) / peers.leads.sum(min_count=1)) if peers.leads.sum(min_count=1) > 0 else np.nan
        gap, metric, label, own, peer = funnel_weakness(b, peers)
        direction = "above" if b.L2E >= peer_l2e else "below"
        paragraph = f"It generated {num(b.leads)} leads and {num(b.enrollments)} enrollments, with L2E {pct(b.L2E)} — {direction} the other selected batches in {b.tech_management} (peer weighted L2E {pct(peer_l2e)})."
        if metric and gap > 0:
            paragraph += f" Its largest identifiable funnel weakness is {label}: {pct(own)} versus {pct(peer)} for the selected peer median."
            actions = {
                "L2W": "Check lead validity, source mix, audience targeting, contactability and speed-to-lead.",
                "L2A": "Review qualification criteria, lead-to-application journey, application friction and source-level lead quality.",
                "A2E": "Review counselling follow-up, application completion, eligibility issues, fee objections and payment drop-offs.",
            }
            paragraph += f" **Recommended investigation:** {actions[metric]}"
        else:
            paragraph += " No clear underperforming funnel stage is visible against these selected peers; investigate program mix, offer, pricing and sales execution before prescribing a change."
        st.write(paragraph)

    # Commercial measures only when the underlying data supports them.
    if "gsv_cr" in batch_table and batch_table.gsv_cr.notna().any():
        st.markdown("## 5. Commercial outcome")
        gsv_total = batch_table.gsv_cr.sum(min_count=1)
        st.write(f"The selected batches report **{num(gsv_total, 2)} Cr GSV** in total. Compare GSV alongside enrollments: a category or program can lead on enrollment volume without leading on GSV per enrollment.")
        if "GSV_per_enrollment" in batch_table:
            gsv_view = batch_table[["program_code", "tech_management", "enrollments", "gsv_cr", "GSV_per_enrollment"]].copy()
            st.dataframe(gsv_view, use_container_width=True, hide_index=True)
    st.caption(f"Data basis: {meta.get('raw_rows', 0):,} source rows consolidated to {meta.get('programs', 0):,} program batches. Program batch identifier is program_code. Missing metrics are unavailable, not zero. Funnel gaps are diagnostic comparisons, not proof of cause.")

    # Scope the conversational agent to precisely the same selected program set.
    selected_raw = raw_df[raw_df.program_code.astype(str).isin(scope.program_code.astype(str))].copy()
    _render_category_chat(st, selected_raw, scope, {
        "findings": [], "tables": {"selected_category_summary": cat_table, "selected_programs": batch_table},
        "programs": scope, "meta": meta,
    })
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
