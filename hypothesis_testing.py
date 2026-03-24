"""
hypothesis_testing.py — Phase 2 group comparison analysis

Loads the cleaned stroke dataset, runs phase2_summary() from analysis_utils,
builds a flat results table, prints it to console, saves it to CSV, prints a
plain-English findings summary, and writes a manuscript-style interpretation file.
"""

import datetime
import pandas as pd
import numpy as np

from analysis_utils import NUMERIC, CATEGORICAL, BINARY_FEATURES, phase2_summary

DATA_PATH            = "data/stroke_data_clean.csv"
OUTPUT_CSV           = "data/phase2_hypothesis_results.csv"
OUTPUT_INTERPRETATION = "data/phase2_interpretation.txt"


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def _sig_flag(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _fmt_median_iqr(grp: dict) -> str:
    return f"{grp['median']:.1f} [{grp['q1']:.1f}-{grp['q3']:.1f}]"


def _fmt_cat_summary(n: int) -> str:
    return f"n={n:,}"


def _fmt_or(or_val: float, lo: float, hi: float) -> str:
    return f"{or_val:.2f} ({lo:.2f}-{hi:.2f})"


# ── Build flat results DataFrame ───────────────────────────────────────────────

def build_results(summary: dict) -> pd.DataFrame:
    rows = []

    for feature, s in summary.items():
        if feature in NUMERIC:
            test        = "Mann-Whitney U"
            stat_val    = round(s["u_stat"], 3)
            p_val       = s["p_value"]
            no_stroke_s = _fmt_median_iqr(s["no_stroke"])
            stroke_s    = _fmt_median_iqr(s["stroke"])
            effect_val  = round(s["cohens_d"], 3)
            effect_lbl  = f"d={effect_val} ({s['magnitude']})"
            or_str      = ""
        else:
            test        = "Chi-square"
            stat_val    = round(s["chi2_stat"], 3)
            p_val       = s["p_value"]
            # Top-level summary: total n per outcome group
            n_no  = sum(v["no_stroke_n"] for v in s["levels"].values())
            n_yes = sum(v["stroke_n"]    for v in s["levels"].values())
            no_stroke_s = _fmt_cat_summary(n_no)
            stroke_s    = _fmt_cat_summary(n_yes)
            effect_val  = round(s["cramers_v"], 3)
            effect_lbl  = f"V={effect_val}"
            or_str      = (
                _fmt_or(s["odds_ratio"], s["or_ci_low"], s["or_ci_high"])
                if feature in BINARY_FEATURES else ""
            )

        rows.append({
            "Feature":          feature,
            "Test":             test,
            "No Stroke":        no_stroke_s,
            "Stroke":           stroke_s,
            "Statistic":        stat_val,
            "p-value":          _fmt_p(p_val),
            "Effect Size":      effect_lbl,
            "Odds Ratio 95% CI": or_str,
            "Sig.":             _sig_flag(p_val),
            "_p_raw":           p_val,          # kept for summary logic, dropped before CSV
        })

    return pd.DataFrame(rows)


# ── Plain-English findings summary ─────────────────────────────────────────────

def print_findings(summary: dict, results: pd.DataFrame) -> None:
    sig_rows = results[results["_p_raw"] < 0.05].copy()

    print("\n" + "=" * 70)
    print("FINDINGS SUMMARY — features significantly associated with stroke")
    print("=" * 70)

    if sig_rows.empty:
        print("No features reached statistical significance (p < 0.05).")
        return

    for _, row in sig_rows.iterrows():
        feat = row["Feature"]
        s    = summary[feat]
        sig  = row["Sig."]

        if feat in NUMERIC:
            ns_med = s["no_stroke"]["median"]
            st_med = s["stroke"]["median"]
            direction = "higher" if st_med > ns_med else "lower"
            magnitude = s["magnitude"]
            print(
                f"  {feat}: {sig} — median {direction} in stroke group "
                f"({st_med:.1f} vs {ns_med:.1f}), effect size {magnitude} "
                f"(d={s['cohens_d']:.3f})"
            )
        elif feat in BINARY_FEATURES:
            or_val = s.get("odds_ratio", np.nan)
            direction = "higher" if or_val > 1 else "lower"
            print(
                f"  {feat}: {sig} — {direction} odds of stroke "
                f"(OR={or_val:.2f}, 95% CI {s['or_ci_low']:.2f}-{s['or_ci_high']:.2f}), "
                f"Cramér's V={s['cramers_v']:.3f}"
            )
        else:
            ref = s["reference_category"]
            print(
                f"  {feat}: {sig} — distribution differs by stroke outcome, "
                f"Cramér's V={s['cramers_v']:.3f} (reference: {ref})"
            )
            for cat, lvl in s["levels"].items():
                if lvl["is_reference"]:
                    continue
                or_val = lvl.get("odds_ratio", np.nan)
                ci_lo  = lvl.get("or_ci_low", np.nan)
                ci_hi  = lvl.get("or_ci_high", np.nan)
                if not np.isnan(or_val):
                    print(
                        f"    {cat} vs {ref}: OR={or_val:.2f} "
                        f"(95% CI {ci_lo:.2f}-{ci_hi:.2f})"
                    )

    print()
    not_sig = results[results["_p_raw"] >= 0.05]["Feature"].tolist()
    if not_sig:
        print(f"Not significant: {', '.join(not_sig)}")
    print()


# ── Manuscript-style interpretation file ───────────────────────────────────────

def _effect_sort_key(row: pd.Series) -> float:
    """Return a numeric effect magnitude for ranking, regardless of test type."""
    es = row["Effect Size"]
    # Numeric features: "d=X.XXX (magnitude)" — extract absolute d value
    if es.startswith("d="):
        try:
            return abs(float(es.split("=")[1].split()[0]))
        except ValueError:
            pass
    # Categorical features: "V=X.XXX" — extract V value
    if es.startswith("V="):
        try:
            return float(es.split("=")[1])
        except ValueError:
            pass
    return 0.0


def _numeric_sentence(feat: str, s: dict) -> str:
    ns   = s["no_stroke"]
    st   = s["stroke"]
    p    = s["p_value"]
    d    = s["cohens_d"]
    mag  = s["magnitude"]
    p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    direction = "higher" if st["median"] > ns["median"] else "lower"
    feat_label = feat.replace("_", " ").replace("avg glucose level", "average glucose level")
    units = {"age": "years", "avg_glucose_level": "mg/dL", "bmi": "kg/m2"}
    unit  = units.get(feat, "")
    unit_str = f" {unit}" if unit else ""
    sentence1 = (
        f"{feat_label.capitalize()} was significantly {direction} in stroke patients "
        f"(median {st['median']:.1f} [IQR {st['q1']:.1f}-{st['q3']:.1f}]{unit_str} vs. "
        f"{ns['median']:.1f} [IQR {ns['q1']:.1f}-{ns['q3']:.1f}]{unit_str}, "
        f"{p_str}, Cohen's d = {d:.3f} — {mag} effect)."
    )
    # Plain-English implication per feature
    implications = {
        "age": (
            "This suggests that older individuals face substantially greater stroke risk, "
            "consistent with the well-established age-related deterioration of vascular health."
        ),
        "avg_glucose_level": (
            "Elevated glucose levels likely reflect chronic hyperglycemia or undiagnosed diabetes, "
            "both of which promote endothelial dysfunction and thrombogenesis."
        ),
        "bmi": (
            "Higher BMI may contribute to stroke risk through its associations with hypertension, "
            "dyslipidemia, and metabolic syndrome, though the effect here is modest."
        ),
    }
    sentence2 = implications.get(feat, "")
    return f"{sentence1}{' ' + sentence2 if sentence2 else ''}"


def _binary_sentence(feat: str, s: dict) -> str:
    or_val = s.get("odds_ratio", np.nan)
    ci_lo  = s.get("or_ci_low", np.nan)
    ci_hi  = s.get("or_ci_high", np.nan)
    p      = s["p_value"]
    p_str  = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    direction = "higher" if or_val > 1 else "lower"
    feat_label = feat.replace("_", " ")
    return (
        f"Patients with {feat_label} had significantly {direction} odds of stroke "
        f"(OR = {or_val:.2f}, 95% CI {ci_lo:.2f}-{ci_hi:.2f}, {p_str}, "
        f"Cramer's V = {s['cramers_v']:.3f})."
    )


def _multilevel_sentence(feat: str, s: dict) -> str:
    p     = s["p_value"]
    v     = s["cramers_v"]
    ref   = s["reference_category"]
    p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    feat_label = feat.replace("_", " ")
    parts = [
        f"The distribution of {feat_label} differed significantly between stroke and "
        f"no-stroke groups ({p_str}, Cramer's V = {v:.3f}; reference: {ref})."
    ]
    for cat, lvl in s["levels"].items():
        if lvl["is_reference"]:
            continue
        ov   = lvl.get("odds_ratio", np.nan)
        lo   = lvl.get("or_ci_low", np.nan)
        hi   = lvl.get("or_ci_high", np.nan)
        if np.isnan(ov):
            continue
        direction = "higher" if ov > 1 else "lower"
        parts.append(
            f"  {cat} vs. {ref}: OR = {ov:.2f} (95% CI {lo:.2f}-{hi:.2f}), "
            f"{direction} odds of stroke."
        )
    return "\n".join(parts)


def _clinical_sentence(feat: str, s: dict) -> str:
    """Return a manuscript-style sentence for a significant feature."""
    if feat in NUMERIC:
        return _numeric_sentence(feat, s)
    if feat in BINARY_FEATURES:
        return _binary_sentence(feat, s)
    return _multilevel_sentence(feat, s)


def write_interpretation(results_df: pd.DataFrame, summary: dict) -> None:
    """Write a plain-English manuscript-style interpretation to OUTPUT_INTERPRETATION."""
    sig   = results_df[results_df["_p_raw"] < 0.05].copy()
    not_sig = results_df[results_df["_p_raw"] >= 0.05]["Feature"].tolist()

    # Rank significant features by effect size descending
    sig["_effect_rank"] = sig.apply(_effect_sort_key, axis=1)
    sig = sig.sort_values("_effect_rank", ascending=False).reset_index(drop=True)

    lines = []

    # ── Header ────────────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines += [
        "Phase 2 - Hypothesis Testing Interpretation",
        f"Generated: {ts}",
        "=" * 70,
        "",
    ]

    # ── Primary findings (top 3 by effect size) ───────────────────────────────
    lines += ["PRIMARY FINDINGS", "-" * 70]
    if sig.empty:
        lines += ["No features reached statistical significance (p < 0.05).", ""]
    else:
        primary = sig.head(3)
        for _, row in primary.iterrows():
            feat = row["Feature"]
            lines += [_clinical_sentence(feat, summary[feat]), ""]

    # ── Secondary findings (remaining significant features) ───────────────────
    if len(sig) > 3:
        lines += ["SECONDARY FINDINGS", "-" * 70]
        secondary = sig.iloc[3:]
        for _, row in secondary.iterrows():
            feat = row["Feature"]
            s    = summary[feat]
            p    = s["p_value"]
            p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            feat_label = feat.replace("_", " ")
            if feat in NUMERIC:
                ns_med = s["no_stroke"]["median"]
                st_med = s["stroke"]["median"]
                direction = "higher" if st_med > ns_med else "lower"
                lines.append(
                    f"{feat_label.capitalize()}: significantly {direction} in stroke patients "
                    f"(median {st_med:.1f} vs. {ns_med:.1f}, {p_str}, "
                    f"Cohen's d = {s['cohens_d']:.3f})."
                )
            elif feat in BINARY_FEATURES:
                or_val = s.get("odds_ratio", np.nan)
                direction = "higher" if or_val > 1 else "lower"
                lines.append(
                    f"{feat_label.capitalize()}: {direction} odds of stroke "
                    f"(OR = {or_val:.2f}, {p_str})."
                )
            else:
                lines.append(
                    f"{feat_label.capitalize()}: distribution differs by stroke outcome "
                    f"({p_str}, Cramer's V = {s['cramers_v']:.3f})."
                )
        lines.append("")

    # ── Non-significant features ───────────────────────────────────────────────
    lines += ["NON-SIGNIFICANT FEATURES", "-" * 70]
    if not_sig:
        feat_list = ", ".join(f.replace("_", " ") for f in not_sig)
        lines += [
            f"The following features did not reach statistical significance (p >= 0.05) "
            f"and showed no meaningful association with stroke outcome in this dataset: "
            f"{feat_list}.",
            "",
        ]
    else:
        lines += ["All tested features reached statistical significance.", ""]

    # ── Limitations ───────────────────────────────────────────────────────────
    lines += [
        "LIMITATIONS",
        "-" * 70,
        (
            "These analyses are based on a cross-sectional observational dataset and therefore "
            "cannot establish causal relationships between risk factors and stroke. "
            "The dataset exhibits marked class imbalance (approximately 5% stroke prevalence), "
            "which may reduce the power to detect true associations in smaller subgroups. "
            "No correction for multiple comparisons (e.g., Bonferroni) was applied; findings "
            "should be interpreted with caution given the number of simultaneous hypothesis "
            "tests performed. Categorical variables were ordinally encoded for the correlation "
            "matrix only and do not imply a true ordinal relationship between categories."
        ),
        "",
    ]

    with open(OUTPUT_INTERPRETATION, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df):,} rows loaded.\n")

    print("Running phase2_summary() ...")
    summary = phase2_summary(df)

    results = build_results(summary)

    # Console output
    display_cols = [c for c in results.columns if c != "_p_raw"]
    print("\n" + results[display_cols].to_string(index=False))

    # Save CSV (drop internal helper column)
    csv_df = results[display_cols]
    csv_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")

    write_interpretation(results, summary)
    print(f"Interpretation saved to {OUTPUT_INTERPRETATION}")

    print_findings(summary, results)


if __name__ == "__main__":
    main()
