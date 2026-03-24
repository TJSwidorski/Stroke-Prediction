"""
hypothesis_testing.py — Phase 2 group comparison analysis

Loads the cleaned stroke dataset, runs phase2_summary() from analysis_utils,
builds a flat results table, prints it to console, saves it to CSV, and
prints a plain-English findings summary.
"""

import pandas as pd
import numpy as np

from analysis_utils import NUMERIC, CATEGORICAL, BINARY_FEATURES, phase2_summary

DATA_PATH  = "data/stroke_data_clean.csv"
OUTPUT_CSV = "data/phase2_hypothesis_results.csv"


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
            print(
                f"  {feat}: {sig} — distribution differs by stroke outcome, "
                f"Cramér's V={s['cramers_v']:.3f}"
            )

    print()
    not_sig = results[results["_p_raw"] >= 0.05]["Feature"].tolist()
    if not_sig:
        print(f"Not significant: {', '.join(not_sig)}")
    print()


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

    print_findings(summary, results)


if __name__ == "__main__":
    main()
