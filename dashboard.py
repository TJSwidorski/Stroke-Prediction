import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from analysis_utils import (
    CATEGORICAL, NUMERIC, ALL_FEATURES, BINARY_FEATURES,
    DEFAULT_BIN_WIDTHS, BIN_WIDTH_OPTIONS,
    feature_stroke_stats, all_features_stats,
    apply_filters, encoded_for_correlation,
    table1_stats, phase2_summary,
)

st.set_page_config(page_title="Stroke Prediction Analysis", layout="wide")
st.title("Stroke Prediction — Data Analysis Dashboard")

DATA_PATH = "data/stroke_data_clean.csv"
STROKE_COLOR = {"No Stroke": "#4C78A8", "Stroke": "#E45756"}
DIR_COLOR = {"Higher ▲": "#E45756", "Lower ▼": "#4C78A8", "—": "#AAAAAA"}
CI_MIN_N = 30  # minimum group size to display confidence interval bars


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["stroke_label"] = df["stroke"].map({0: "No Stroke", 1: "Stroke"})
    return df


@st.cache_data
def load_all_stats():
    return all_features_stats(load_data())


@st.cache_data
def load_table1():
    return table1_stats(load_data())


@st.cache_data
def load_phase2_detail():
    return phase2_summary(load_data())


try:
    df = load_data()
except FileNotFoundError:
    st.error("Data not found. Run `retrieve_data.py` then `feature_engineering.py` first.")
    st.stop()

p0 = df["stroke"].mean()
overall_rate = p0 * 100

tab0, tab_t1, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏥 Cohort Overview",
    "📋 Table 1",
    "📊 Distributions",
    "🎯 Individual Stroke Risk",
    "🔗 Joint Stroke Risk",
    "🔥 Correlation Matrix",
    "🧪 Hypothesis Testing",
    "📊 Phase 2 — Group Comparison",
])


# ── Tab 0: Cohort Overview ────────────────────────────────────────────────────
with tab0:
    st.header("Cohort Overview")

    n_total = len(df)
    n_stroke = int(df["stroke"].sum())
    n_no_stroke = n_total - n_stroke
    prevalence = overall_rate

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Patients", f"{n_total:,}")
    m2.metric("Stroke Cases", f"{n_stroke:,}")
    m3.metric("Non-Stroke Cases", f"{n_no_stroke:,}")
    m4.metric("Stroke Prevalence", f"{prevalence:.1f}%")

    st.markdown(
        f"**Note:** stroke cases represent only **{prevalence:.1f}%** of the cohort — "
        "a severe class imbalance that affects model evaluation and will be addressed with SMOTE."
    )

    with st.expander("Data quality summary"):
        try:
            with open("data/data_quality_report.txt") as f:
                report_text = f.read()
            st.code(report_text, language=None)
        except FileNotFoundError:
            st.warning('Run `feature_engineering.py` to generate the data quality report.')

    with st.expander("Outlier flags"):
        outlier_cols = {
            "age":               "age_outlier",
            "avg_glucose_level": "glucose_outlier",
            "bmi":               "bmi_outlier",
        }
        outlier_rows = []
        for col, flag_col in outlier_cols.items():
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            flagged = int(df[flag_col].sum())
            outlier_rows.append({
                "Variable": col,
                "IQR Lower Bound": round(lower, 2),
                "IQR Upper Bound": round(upper, 2),
                "Flagged Count": flagged,
                "Flagged %": round(flagged / n_total * 100, 1),
            })
        st.dataframe(
            pd.DataFrame(outlier_rows).set_index("Variable"),
            use_container_width=True,
        )


# ── Tab T1: Table 1 ───────────────────────────────────────────────────────────
with tab_t1:
    st.header("Table 1 — Cohort Characteristics by Stroke Outcome")

    n_no  = int((df["stroke"] == 0).sum())
    n_yes = int((df["stroke"] == 1).sum())

    col_no  = f"No Stroke (n={n_no:,})"
    col_yes = f"Stroke (n={n_yes:,}, {overall_rate:.1f}%)"

    t1 = load_table1()

    display = t1[["variable", "no_stroke_fmt", "stroke_fmt", "p_value_fmt"]].copy()
    display.columns = ["Variable", col_no, col_yes, "p-value"]

    def _highlight_sig(val):
        if isinstance(val, str) and val.endswith("*"):
            return "font-weight: bold; color: #c0392b"
        return ""

    styled = display.style.map(_highlight_sig, subset=["p-value"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.caption(
        "Continuous variables reported as Mean (SD). "
        "P-values from Mann-Whitney U test (continuous) and chi-square test (categorical). "
        "* p < 0.05."
    )

    csv_bytes = display.to_csv(index=False).encode()
    st.download_button(
        label="⬇ Download Table 1 as CSV",
        data=csv_bytes,
        file_name="table1.csv",
        mime="text/csv",
    )


# ── Tab 1: Distributions ──────────────────────────────────────────────────────
with tab1:
    st.header("Feature Distributions")

    with st.expander("ℹ️ How to read these charts"):
        st.markdown(
            "**Categorical features** show a stacked percentage bar: each bar sums to 100%, "
            "so you can compare stroke prevalence across categories regardless of group size.\n\n"
            "**Numeric features** show two charts:\n"
            "- A **count histogram** (optionally split by outcome) to see where patients are concentrated.\n"
            "- A **box plot** split by outcome to compare the central tendency and spread between "
            "stroke and non-stroke patients.\n\n"
            "For rates and significance testing, see **Individual Stroke Risk**."
        )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        feature = st.selectbox("Feature", ALL_FEATURES, key="dist_feat")
    with col_b:
        if feature in NUMERIC:
            split = st.checkbox("Split by stroke outcome", value=True)
        else:
            split = True
            st.empty()

    if feature in CATEGORICAL:
        counts = (
            df.groupby([feature, "stroke_label"])
            .size()
            .reset_index(name="count")
        )
        counts["pct"] = (
            counts["count"]
            / counts.groupby(feature)["count"].transform("sum")
            * 100
        )
        fig = px.bar(
            counts, x=feature, y="pct", color="stroke_label",
            barmode="stack",
            color_discrete_map=STROKE_COLOR,
            labels={"pct": "Patients (%)", "stroke_label": "Outcome"},
            category_orders={"stroke_label": ["No Stroke", "Stroke"]},
        )
        fig.update_layout(legend_title_text="Outcome", yaxis_title="Patients (%)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Each bar shows the percentage breakdown of stroke vs. no-stroke patients "
            "within that category, removing raw count bias."
        )
    else:
        fig = px.histogram(
            df, x=feature,
            color="stroke_label" if split else None,
            barmode="overlay",
            opacity=0.7,
            nbins=30,
            color_discrete_map=STROKE_COLOR,
            labels={"stroke_label": "Outcome"},
            category_orders={"stroke_label": ["No Stroke", "Stroke"]},
        )
        fig.update_layout(legend_title_text="Outcome")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Count of patients across the selected numeric feature"
            + (" split by stroke outcome." if split else ".")
        )

        fig_box = px.box(
            df, x="stroke_label", y=feature,
            color="stroke_label",
            color_discrete_map=STROKE_COLOR,
            labels={"stroke_label": "Outcome", feature: feature},
            category_orders={"stroke_label": ["No Stroke", "Stroke"]},
        )
        fig_box.update_layout(showlegend=False, xaxis_title="Outcome")
        st.plotly_chart(fig_box, use_container_width=True)
        st.caption(
            "Box plot comparing the median, spread, and outliers of the selected feature "
            "between stroke and non-stroke patients."
        )


# ── Tab 2: Individual Stroke Risk ─────────────────────────────────────────────
with tab2:
    st.header("Individual Feature Stroke Risk")

    with st.expander("ℹ️ How to read this chart"):
        st.markdown(
            "Shows the **stroke rate (%)** — the proportion of patients who had a stroke — "
            "within each value or bin of the selected feature.\n\n"
            f"- **Dashed line** = overall population stroke rate ({overall_rate:.1f}%), the baseline for comparison.\n"
            f"- **Error bars** = 95% Agresti-Coull confidence interval on the observed rate. "
            f"Only shown for groups with N ≥ {CI_MIN_N}; smaller groups are omitted to avoid "
            f"misleadingly wide bars.\n"
            "- For numeric features, use the **Bin size** selector to control the width of each "
            "group (e.g. 10-year age gaps vs. 1-year gaps).\n"
            "- **Bar color** reflects the hypothesis test result (α = 0.05):\n"
            "  - 🔴 Red — stroke rate is *significantly higher* than the overall rate\n"
            "  - 🔵 Blue — stroke rate is *significantly lower* than the overall rate\n"
            "  - ⬜ Gray — not significantly different from the overall rate\n\n"
            "See **Hypothesis Testing** for full test details."
        )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        feature2 = st.selectbox("Feature", ALL_FEATURES, key="risk_feat")
    with col_b:
        if feature2 in NUMERIC:
            bin_width2 = st.select_slider(
                "Bin size",
                options=BIN_WIDTH_OPTIONS[feature2],
                value=DEFAULT_BIN_WIDTHS[feature2],
                key="risk_bins",
            )
        else:
            bin_width2 = None
            st.empty()

    stats_df = feature_stroke_stats(df, feature2, bin_width=bin_width2)

    ci_upper_arr = [
        (row["ci_upper"] - row["rate"]) * 100 if row["n"] >= CI_MIN_N else None
        for _, row in stats_df.iterrows()
    ]
    ci_lower_arr = [
        (row["rate"] - row["ci_lower"]) * 100 if row["n"] >= CI_MIN_N else None
        for _, row in stats_df.iterrows()
    ]

    def hover_ci(row):
        if row["n"] >= CI_MIN_N:
            return f"95% CI: [{row['ci_lower'] * 100:.1f}%, {row['ci_upper'] * 100:.1f}%]"
        return f"CI not shown (N={row['n']} < {CI_MIN_N})"

    fig2 = go.Figure(go.Bar(
        x=stats_df["label"],
        y=stats_df["rate"] * 100,
        error_y=dict(
            type="data",
            symmetric=False,
            array=ci_upper_arr,
            arrayminus=ci_lower_arr,
            thickness=2,
            width=6,
        ),
        marker_color=[DIR_COLOR[d] for d in stats_df["direction"]],
        text=stats_df["rate"].apply(lambda r: f"{r * 100:.1f}%"),
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Stroke rate: %{y:.1f}%<br>"
            "%{customdata[0]}<br>"
            "p-value: %{customdata[1]}<br>"
            "%{customdata[2]}<extra></extra>"
        ),
        customdata=list(zip(
            [hover_ci(row) for _, row in stats_df.iterrows()],
            stats_df["p_value"].apply(lambda p: f"{p:.4f}" if not np.isnan(p) else "—"),
            stats_df["sig"],
        )),
    ))
    fig2.add_hline(
        y=overall_rate, line_dash="dash", line_color="gray",
        annotation_text=f"Overall: {overall_rate:.1f}%",
        annotation_position="top left",
    )
    fig2.update_layout(
        xaxis_title=feature2,
        yaxis_title="Stroke Rate (%)",
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

    display_df = stats_df[["label", "n", "strokes", "rate", "ci_lower", "ci_upper",
                            "z_stat", "p_value", "sig", "direction"]].copy()
    display_df["rate"] = (display_df["rate"] * 100).round(2)
    display_df["ci_lower"] = (display_df["ci_lower"] * 100).round(2)
    display_df["ci_upper"] = (display_df["ci_upper"] * 100).round(2)
    display_df["z_stat"] = display_df["z_stat"].round(3)
    display_df["p_value"] = display_df["p_value"].apply(lambda p: f"{p:.4f}" if not np.isnan(p) else "—")
    display_df.columns = ["Value", "N", "Strokes", "Rate (%)", "CI Lower (%)",
                          "CI Upper (%)", "Z-stat", "p-value", "Sig.", "Direction"]
    st.dataframe(display_df.set_index("Value"), use_container_width=True)


# ── Tab 3: Joint Stroke Risk ──────────────────────────────────────────────────
with tab3:
    st.header("Joint Stroke Risk")

    with st.expander("ℹ️ How to read this chart"):
        st.markdown(
            "Computes the **conditional stroke probability** P(stroke | selected conditions) — "
            "the observed stroke rate among only the patients who match all chosen filters.\n\n"
            "- Use this to estimate stroke risk for a specific patient profile "
            "(e.g., elderly + hypertensive + smoker).\n"
            "- The **gauge needle** shows the filtered group's rate; the **blue threshold line** "
            f"marks the overall population rate ({overall_rate:.1f}%) for comparison.\n"
            "- Narrow your filters gradually. Very specific combinations may match few patients, "
            "making the estimate noisy."
        )

    selected = st.multiselect("Features to filter on", ALL_FEATURES)

    filters = {}
    if selected:
        cols = st.columns(min(len(selected), 3))
        for i, feat in enumerate(selected):
            with cols[i % 3]:
                if feat in CATEGORICAL:
                    options = sorted(df[feat].unique().tolist())
                    val = st.selectbox(feat, options, key=f"j_{feat}")
                    filters[feat] = ("eq", val)
                else:
                    lo, hi = float(df[feat].min()), float(df[feat].max())
                    st.caption(feat)
                    lo_val = st.number_input(
                        "Min", min_value=lo, max_value=hi, value=lo,
                        key=f"j_{feat}_lo",
                    )
                    hi_val = st.number_input(
                        "Max", min_value=lo, max_value=hi, value=hi,
                        key=f"j_{feat}_hi",
                    )
                    filters[feat] = ("range", (lo_val, hi_val))

        filtered = apply_filters(df, filters)
        n = len(filtered)
        rate = filtered["stroke"].mean() * 100 if n > 0 else 0.0

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Patients matching filters", f"{n:,}")
        m2.metric("Stroke rate", f"{rate:.1f}%")
        m3.metric("vs. overall rate", f"{rate - overall_rate:+.1f}%")

        if n < CI_MIN_N:
            st.warning(f"Only {n} patients match these filters — estimate may be unreliable.")

        fig3 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rate,
            number={"suffix": "%", "font": {"size": 40}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#E45756"},
                "steps": [{"range": [0, 100], "color": "#f0f0f0"}],
                "threshold": {
                    "line": {"color": "#4C78A8", "width": 3},
                    "thickness": 0.75,
                    "value": overall_rate,
                },
            },
            title={"text": f"Stroke Probability<br><sup>Blue line = overall {overall_rate:.1f}%</sup>"},
        ))
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)

        if n > 0:
            with st.expander("View matching patients"):
                st.dataframe(filtered.drop(columns=["stroke_label"]), use_container_width=True)


# ── Tab 4: Correlation Matrix ─────────────────────────────────────────────────
with tab4:
    st.header("Correlation Matrix")

    with st.expander("ℹ️ How to read this chart"):
        st.markdown(
            "Shows **Pearson correlation coefficients** between every pair of features, "
            "ranging from −1 (perfect inverse relationship) to +1 (perfect positive relationship). "
            "Values near 0 indicate little linear association.\n\n"
            "- Categorical features are **ordinally encoded** as integers for this calculation. "
            "The ordering is arbitrary for nominal categories (e.g. work type), "
            "so treat those correlations as approximate.\n"
            "- High correlation with **stroke** suggests predictive value for the model.\n"
            "- High correlation *between* features may indicate redundancy (collinearity), "
            "which can affect model training."
        )

    df_enc = encoded_for_correlation(df)
    corr = df_enc.corr()

    fig4 = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
    )
    fig4.update_layout(height=600)
    st.plotly_chart(fig4, use_container_width=True)

    with st.expander("Correlations with stroke (sorted)"):
        stroke_corr = corr["stroke"].drop("stroke").sort_values(key=abs, ascending=False)
        st.dataframe(
            stroke_corr.rename("Correlation with Stroke").to_frame(),
            use_container_width=True,
        )


# ── Tab 5: Hypothesis Testing ─────────────────────────────────────────────────
with tab5:
    st.header("Hypothesis Testing — Single Proportion")

    with st.expander("ℹ️ How to read these results"):
        st.markdown(
            f"For each feature value or bin, we test whether its stroke rate differs "
            f"from the overall population rate (p₀ = **{p0:.4f}**, i.e. {overall_rate:.2f}%).\n\n"
            "**Null hypothesis H₀:** the stroke rate for this group equals p₀\n\n"
            "**Test:** two-sided z-test for a single proportion (α = 0.05). "
            "The z-statistic measures how many standard errors the observed rate is from p₀ — "
            "values beyond ±1.96 cross the significance threshold (dashed lines on the chart).\n\n"
            "**Confidence intervals** use the Agresti-Coull method, which is more accurate than "
            "the standard Wald interval, particularly for small samples or extreme proportions. "
            f"CIs are only displayed for groups with N ≥ {CI_MIN_N}.\n\n"
            "**Significance stars:** `***` p < 0.001 · `**` p < 0.01 · `*` p < 0.05 · `ns` not significant"
        )

    all_stats = load_all_stats()
    sig_df = all_stats[all_stats["significant"]]
    n_higher = (sig_df["direction"] == "Higher ▲").sum()
    n_lower = (sig_df["direction"] == "Lower ▼").sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Feature values tested", len(all_stats))
    m2.metric("Significant results", len(sig_df))
    m3.metric("Significantly higher ▲", int(n_higher))
    m4.metric("Significantly lower ▼", int(n_lower))

    st.divider()

    # ── General findings ──────────────────────────────────────────────────────
    st.subheader("General Findings")
    st.caption(
        "Each bar shows the z-statistic for one feature value. "
        "Bars beyond the ±1.96 dashed lines are statistically significant. "
        "Red bars indicate higher stroke risk than the population; blue bars indicate lower risk."
    )

    all_stats_sorted = all_stats.sort_values("z_stat", ascending=True)
    fig_z = go.Figure(go.Bar(
        x=all_stats_sorted["z_stat"],
        y=all_stats_sorted["feature"] + " · " + all_stats_sorted["label"],
        orientation="h",
        marker_color=[DIR_COLOR[d] for d in all_stats_sorted["direction"]],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Z-stat: %{x:.3f}<br>"
            "p-value: %{customdata[0]}<br>"
            "Stroke rate: %{customdata[1]:.1f}%<br>"
            "N: %{customdata[2]}<br>"
            "%{customdata[3]}<extra></extra>"
        ),
        customdata=list(zip(
            all_stats_sorted["p_value"].apply(lambda p: f"{p:.4f}"),
            all_stats_sorted["rate"] * 100,
            all_stats_sorted["n"],
            all_stats_sorted["sig"],
        )),
    ))
    fig_z.add_vline(x=0, line_color="gray", line_width=1)
    fig_z.add_vline(x=1.96, line_dash="dash", line_color="#888",
                    annotation_text="α = 0.05", annotation_position="top")
    fig_z.add_vline(x=-1.96, line_dash="dash", line_color="#888")
    fig_z.update_layout(
        xaxis_title="Z-statistic  (positive = stroke rate above p₀, negative = below p₀)",
        yaxis_title=None,
        height=max(400, len(all_stats) * 22),
        showlegend=False,
    )
    st.plotly_chart(fig_z, use_container_width=True)

    if len(sig_df) > 0:
        with st.expander("Significant results table", expanded=True):
            display_sig = sig_df[["feature", "label", "n", "strokes", "rate",
                                   "ci_lower", "ci_upper", "z_stat", "p_value", "sig", "direction"]].copy()
            display_sig["rate"] = (display_sig["rate"] * 100).round(2)
            display_sig["ci_lower"] = (display_sig["ci_lower"] * 100).round(2)
            display_sig["ci_upper"] = (display_sig["ci_upper"] * 100).round(2)
            display_sig["z_stat"] = display_sig["z_stat"].round(3)
            display_sig["p_value"] = display_sig["p_value"].apply(lambda p: f"{p:.4f}")
            display_sig.columns = ["Feature", "Value", "N", "Strokes", "Rate (%)",
                                    "CI Lower (%)", "CI Upper (%)", "Z-stat", "p-value", "Sig.", "Direction"]
            st.dataframe(
                display_sig.sort_values("p-value").set_index("Feature"),
                use_container_width=True,
            )

    st.divider()

    # ── Feature detail ────────────────────────────────────────────────────────
    st.subheader("Feature Detail")
    st.caption(
        "Select a feature to see a forest plot — each point is one value/bin, "
        "with whiskers showing the 95% Agresti-Coull CI (N ≥ 30 only). "
        "The dashed line marks p₀. Points right of it have higher stroke rates; left have lower."
    )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        detail_feat = st.selectbox("Select a feature group", ALL_FEATURES, key="hyp_feat")
    with col_b:
        if detail_feat in NUMERIC:
            bin_width5 = st.select_slider(
                "Bin size",
                options=BIN_WIDTH_OPTIONS[detail_feat],
                value=DEFAULT_BIN_WIDTHS[detail_feat],
                key="hyp_bins",
            )
        else:
            bin_width5 = None
            st.empty()

    feat_stats = feature_stroke_stats(df, detail_feat, bin_width=bin_width5)

    fig_forest = go.Figure()
    for direction, color in DIR_COLOR.items():
        subset = feat_stats[feat_stats["direction"] == direction]
        if subset.empty:
            continue

        ci_upper_arr = [
            (row["ci_upper"] - row["rate"]) * 100 if row["n"] >= CI_MIN_N else None
            for _, row in subset.iterrows()
        ]
        ci_lower_arr = [
            (row["rate"] - row["ci_lower"]) * 100 if row["n"] >= CI_MIN_N else None
            for _, row in subset.iterrows()
        ]

        fig_forest.add_trace(go.Scatter(
            x=subset["rate"] * 100,
            y=subset["label"],
            mode="markers",
            name=direction if direction != "—" else "Not significant",
            marker=dict(color=color, size=11, symbol="circle"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=ci_upper_arr,
                arrayminus=ci_lower_arr,
                color=color,
                thickness=2,
                width=8,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Stroke rate: %{x:.1f}%<br>"
                "%{customdata[0]}<br>"
                "Z-stat: %{customdata[1]:.3f}<br>"
                "p-value: %{customdata[2]}<br>"
                "N: %{customdata[3]}<br>"
                "%{customdata[4]}<extra></extra>"
            ),
            customdata=list(zip(
                [hover_ci(row) for _, row in subset.iterrows()],
                subset["z_stat"],
                subset["p_value"].apply(lambda p: f"{p:.4f}"),
                subset["n"],
                subset["sig"],
            )),
        ))

    fig_forest.add_vline(
        x=overall_rate, line_dash="dash", line_color="gray",
        annotation_text=f"p₀ = {overall_rate:.1f}%",
        annotation_position="top right",
    )
    fig_forest.update_layout(
        xaxis_title="Stroke Rate (%)",
        yaxis_title=detail_feat,
        height=max(300, len(feat_stats) * 55),
        legend_title_text="Direction",
    )
    st.plotly_chart(fig_forest, use_container_width=True)

    detail_display = feat_stats[["label", "n", "strokes", "rate", "ci_lower", "ci_upper",
                                  "z_stat", "p_value", "sig", "direction"]].copy()
    detail_display["rate"] = (detail_display["rate"] * 100).round(2)
    detail_display["ci_lower"] = (detail_display["ci_lower"] * 100).round(2)
    detail_display["ci_upper"] = (detail_display["ci_upper"] * 100).round(2)
    detail_display["z_stat"] = detail_display["z_stat"].round(3)
    detail_display["p_value"] = detail_display["p_value"].apply(
        lambda p: f"{p:.4f}" if not np.isnan(p) else "—"
    )
    detail_display.columns = ["Value", "N", "Strokes", "Rate (%)", "CI Lower (%)",
                               "CI Upper (%)", "Z-stat", "p-value", "Sig.", "Direction"]
    st.dataframe(detail_display.set_index("Value"), use_container_width=True)


# ── Tab 6: Phase 2 — Group Comparison ────────────────────────────────────────
PHASE2_CSV = "data/phase2_hypothesis_results.csv"

with tab6:
    st.header("Phase 2 — Group Comparison (Stroke vs. No Stroke)")

    with st.expander("ℹ️ How this differs from the Hypothesis Testing tab"):
        st.markdown(
            "**Hypothesis Testing tab** asks: *Is this group's stroke rate different from the "
            "overall population average?* It compares each subgroup (e.g. smokers) against the "
            "whole dataset's stroke rate. This is useful for spotting which values stand out, "
            "but it doesn't directly compare stroke patients to non-stroke patients.\n\n"
            "**This tab** asks: *Do stroke patients and non-stroke patients look different on "
            "this feature?* It puts the two groups head-to-head — 249 stroke patients vs. "
            "4,861 non-stroke patients — and tests whether their distributions differ. "
            "This is the standard approach in clinical research for identifying risk factors.\n\n"
            "- **Continuous features** (age, glucose, BMI): Mann-Whitney U test — does not "
            "assume the data is normally distributed, which matters here given the skewed "
            "distributions. Effect size is Cohen's d.\n"
            "- **Categorical features** (gender, work type, smoking status): Chi-square test — "
            "checks whether the mix of categories differs between groups. Effect size is "
            "Cramér's V (0 = no association, 1 = perfect association). Per-level **odds ratios** "
            "are reported using **dummy coding**: each category is compared to a reference "
            "category (the first alphabetically), so you can see which specific levels drive "
            "the overall chi-square result.\n"
            "- **Binary features** (hypertension, heart disease, ever married, residence type): "
            "also report a single **odds ratio** — how many times more (or less) likely a "
            "patient with that characteristic is to have had a stroke, compared to the "
            "other value of that feature."
        )

    # ── Load results CSV ──────────────────────────────────────────────────────
    try:
        p2 = pd.read_csv(PHASE2_CSV)
    except FileNotFoundError:
        p2 = None
        st.warning("Run `hypothesis_testing.py` to generate Phase 2 results.")

    if p2 is not None:
        # ── Helper: parse formatted p-value string to float ──────────────────
        def _parse_p(s: str) -> float:
            if isinstance(s, str) and s.strip() == "< 0.001":
                return 0.0005
            try:
                return float(s)
            except (ValueError, TypeError):
                return np.nan

        p2["_p_raw"] = p2["p-value"].apply(_parse_p)

        # ── Metric cards ─────────────────────────────────────────────────────
        n_total   = len(p2)
        n_sig     = int((p2["_p_raw"] < 0.05).sum())
        n_large   = int(p2["Effect Size"].str.contains("large", na=False).sum())
        or_vals   = (
            p2["Odds Ratio 95% CI"]
            .dropna()
            .loc[p2["Odds Ratio 95% CI"].astype(str).str.strip() != ""]
            .apply(lambda s: float(s.split()[0]) if s else np.nan)
            .dropna()
        )
        n_or_high = int((or_vals > 2.0).sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Features tested",        n_total)
        m2.metric("Significant (p < 0.05)", n_sig)
        m3.metric("Large effect size",       n_large)
        m4.metric("Odds ratio > 2.0",        n_or_high if n_or_high > 0 else "N/A")

        st.divider()

        # ── Results table ─────────────────────────────────────────────────────
        SIG_COLORS = {"***": "#c0392b", "**": "#e67e22", "*": "#b8860b", "ns": "#888888"}

        def _color_sig(val):
            color = SIG_COLORS.get(str(val).strip(), "#888888")
            return f"color: {color}; font-weight: bold"

        display_cols = [c for c in p2.columns if c != "_p_raw"]
        styled_p2 = p2[display_cols].style.map(_color_sig, subset=["Sig."])
        st.dataframe(styled_p2, use_container_width=True, hide_index=True)

        csv_bytes = p2[display_cols].to_csv(index=False).encode()
        st.download_button(
            label="⬇ Download Phase 2 results",
            data=csv_bytes,
            file_name="phase2_hypothesis_results.csv",
            mime="text/csv",
        )

        # ── Interpretations expander ──────────────────────────────────────────
        with st.expander("📝 Interpretations"):
            p2_det = load_phase2_detail()

            def _or_sentence(or_val, ci_lo, ci_hi, subj, ref_desc, p_raw):
                sig_note = "" if p_raw < 0.05 else " (not statistically significant)"
                direction = "higher" if or_val >= 1.0 else "lower"
                return (
                    f"{subj} had **{or_val:.2f}× {direction} odds** of stroke "
                    f"compared to {ref_desc} "
                    f"(OR = {or_val:.2f}, 95% CI {ci_lo:.2f}–{ci_hi:.2f}){sig_note}."
                )

            for feat in CATEGORICAL:
                s       = p2_det[feat]
                p_raw   = s["p_value"]
                ref     = s["reference_category"]
                levels  = s["levels"]
                all_cats = sorted(levels.keys())

                st.markdown(f"**{feat}**")

                if feat in BINARY_FEATURES:
                    # Single top-level sentence for the active (second) category vs reference
                    active = all_cats[1]
                    subj     = f"Patients with **{feat} = {active}**"
                    ref_desc = f"those with **{feat} = {ref}**"
                    st.markdown(
                        "- " + _or_sentence(
                            s["odds_ratio"], s["or_ci_low"], s["or_ci_high"],
                            subj, ref_desc, p_raw,
                        )
                    )
                else:
                    if p_raw >= 0.05:
                        st.markdown(
                            f"- No significant difference in stroke outcome across "
                            f"{feat} categories (p = {s['p_value']:.3f})."
                        )
                    else:
                        for cat in all_cats:
                            lvl = levels[cat]
                            if lvl["is_reference"]:
                                st.markdown(
                                    f"- **{cat}** — reference category (OR = 1.00 by definition)."
                                )
                            else:
                                or_val = lvl.get("odds_ratio", np.nan)
                                if np.isnan(or_val):
                                    st.markdown(f"- **{cat}** — OR could not be computed (likely zero-cell).")
                                else:
                                    subj     = f"**{cat}** patients"
                                    ref_desc = f"**{ref}** patients"
                                    st.markdown(
                                        "- " + _or_sentence(
                                            or_val, lvl["or_ci_low"], lvl["or_ci_high"],
                                            subj, ref_desc, p_raw,
                                        )
                                    )

        st.divider()

        # ── Feature detail ────────────────────────────────────────────────────
        st.subheader("Feature Detail")

        detail_feat_p2 = st.selectbox(
            "Select a feature", p2["Feature"].tolist(), key="p2_feat"
        )
        feat_row = p2[p2["Feature"] == detail_feat_p2].iloc[0]
        p_fmt    = feat_row["p-value"]
        sig_flag = feat_row["Sig."]

        if detail_feat_p2 in NUMERIC:
            fig_p2 = px.box(
                df, x="stroke_label", y=detail_feat_p2,
                color="stroke_label",
                color_discrete_map=STROKE_COLOR,
                labels={"stroke_label": "Outcome", detail_feat_p2: detail_feat_p2},
                category_orders={"stroke_label": ["No Stroke", "Stroke"]},
            )
            fig_p2.update_layout(showlegend=False, xaxis_title="Outcome")
            st.plotly_chart(fig_p2, use_container_width=True)
            st.caption(
                f"Mann-Whitney U test: p = {p_fmt} ({sig_flag})  |  "
                f"Effect size: {feat_row['Effect Size']}"
            )
        else:
            feat_stats_p2 = feature_stroke_stats(df, detail_feat_p2)
            fig_p2 = px.bar(
                feat_stats_p2, x="label", y="rate",
                color="direction",
                color_discrete_map={
                    "Higher ▲": STROKE_COLOR["Stroke"],
                    "Lower ▼":  STROKE_COLOR["No Stroke"],
                    "—":        "#AAAAAA",
                },
                labels={"label": detail_feat_p2, "rate": "Stroke Rate"},
                text=feat_stats_p2["rate"].apply(lambda r: f"{r * 100:.1f}%"),
            )
            fig_p2.update_traces(textposition="outside")
            fig_p2.update_layout(
                yaxis_tickformat=".0%",
                yaxis_title="Stroke Rate",
                showlegend=False,
            )
            st.plotly_chart(fig_p2, use_container_width=True)
            st.caption(
                f"Chi-square test: p = {p_fmt} ({sig_flag})  |  "
                f"Effect size: {feat_row['Effect Size']}"
                + (f"  |  OR: {feat_row['Odds Ratio 95% CI']}"
                   if str(feat_row.get("Odds Ratio 95% CI", "")).strip() else "")
            )

            # Per-level OR table (dummy coding vs reference category)
            p2_detail  = load_phase2_detail()
            feat_s     = p2_detail[detail_feat_p2]
            ref_cat    = feat_s["reference_category"]
            or_rows = []
            for cat, lvl in feat_s["levels"].items():
                label = f"{cat} (ref)" if lvl["is_reference"] else str(cat)
                if lvl["is_reference"]:
                    or_str = "1.00 (reference)"
                else:
                    ov = lvl.get("odds_ratio", np.nan)
                    or_str = (
                        f"{ov:.2f} ({lvl['or_ci_low']:.2f}-{lvl['or_ci_high']:.2f})"
                        if not np.isnan(ov) else "—"
                    )
                or_rows.append({
                    "Category":         label,
                    "No Stroke n (%)":  f"{lvl['no_stroke_n']} ({lvl['no_stroke_rate']:.1f}%)",
                    "Stroke n (%)":     f"{lvl['stroke_n']} ({lvl['stroke_rate']:.1f}%)",
                    "OR (95% CI)":      or_str,
                })
            st.dataframe(pd.DataFrame(or_rows), hide_index=True, use_container_width=True)
            st.caption(
                f"Odds ratios computed vs reference category '{ref_cat}' (dummy coding). "
                "OR > 1 indicates higher odds of stroke relative to the reference."
            )

        st.divider()

        # ── Odds Ratio Forest Plot ────────────────────────────────────────────
        st.subheader("Odds Ratio Forest Plot")

        with st.expander("ℹ️ What is an odds ratio?"):
            st.markdown(
                "An **odds ratio (OR)** measures how much more (or less) likely a patient "
                "with a given characteristic is to have had a stroke, compared to a reference "
                "group.\n\n"
                "- **OR = 1.0** — no difference in stroke odds between the two groups "
                "(the dashed reference line on this chart).\n"
                "- **OR > 1.0** — the group has *higher* odds of stroke than the reference. "
                "For example, OR = 3.70 for hypertension means hypertensive patients had "
                "3.7× the stroke odds of non-hypertensive patients.\n"
                "- **OR < 1.0** — the group has *lower* odds of stroke than the reference. "
                "For example, OR = 0.06 for 'children' work type means children had 94% "
                "lower stroke odds than Private-sector workers.\n\n"
                "**Binary features** (hypertension, heart disease, ever married, residence "
                "type) compare the two values of that feature directly.\n\n"
                "**Multi-level features** (gender, work type, smoking status) use **dummy "
                "coding**: each category is compared to a chosen reference category "
                "(shown as a ◆ diamond at OR = 1.0). Reference categories are: "
                "work type → *Private*, smoking status → *never smoked*, "
                "gender → *Female*.\n\n"
                "**Confidence intervals (error bars):** if the 95% CI crosses the OR = 1.0 "
                "line, the result is not statistically significant. "
                "**Color:** red = significantly higher risk, blue = significantly lower "
                "risk, gray = not significant or reference."
            )

        def _parse_or_ci(s: str):
            parts  = s.strip().split()
            or_val = float(parts[0])
            ci_str = parts[1].strip("()")
            lo, hi = ci_str.split("-")
            return float(or_val), float(lo), float(hi)

        # ── Build combined forest plot rows ───────────────────────────────────
        p2_det_fp      = load_phase2_detail()
        non_binary_cats = [f for f in CATEGORICAL if f not in BINARY_FEATURES]
        binary_cats     = [f for f in CATEGORICAL if f in BINARY_FEATURES]

        fp_rows = []

        # Multi-level categoricals — per-level dummy-coded ORs (appear at bottom)
        for feat in non_binary_cats:
            s   = p2_det_fp[feat]
            ref = s["reference_category"]
            # Reference row (diamond at 1.0)
            fp_rows.append({
                "y": f"{feat}: {ref}", "or": 1.0,
                "lo": np.nan, "hi": np.nan,
                "color": DIR_COLOR["—"], "symbol": "diamond",
                "label": "(reference)", "group_ord": CATEGORICAL.index(feat),
                "within_ord": 0,
            })
            for cat, lvl in s["levels"].items():
                if lvl["is_reference"]:
                    continue
                ov = lvl.get("odds_ratio", np.nan)
                if np.isnan(ov):
                    continue  # skip zero-cell levels
                lo, hi = lvl["or_ci_low"], lvl["or_ci_high"]
                color = (
                    DIR_COLOR["Higher ▲"] if lo > 1.0
                    else DIR_COLOR["Lower ▼"] if hi < 1.0
                    else DIR_COLOR["—"]
                )
                fp_rows.append({
                    "y": f"{feat}: {cat}", "or": ov,
                    "lo": lo, "hi": hi,
                    "color": color, "symbol": "circle",
                    "label": f"{ov:.2f} ({lo:.2f}-{hi:.2f})",
                    "group_ord": CATEGORICAL.index(feat), "within_ord": ov,
                })

        # Binary features — top-level OR (appear at top)
        or_mask = (
            p2["Odds Ratio 95% CI"].notna() &
            (p2["Odds Ratio 95% CI"].astype(str).str.strip() != "")
        )
        for _, row in p2[or_mask].iterrows():
            ov, lo, hi = _parse_or_ci(row["Odds Ratio 95% CI"])
            sig = str(row["Sig."]).strip()
            color = (
                DIR_COLOR["—"] if sig == "ns"
                else DIR_COLOR["Higher ▲"] if ov > 1.0
                else DIR_COLOR["Lower ▼"]
            )
            fp_rows.append({
                "y": row["Feature"], "or": ov,
                "lo": lo, "hi": hi,
                "color": color, "symbol": "circle",
                "label": row["Odds Ratio 95% CI"],
                "group_ord": len(CATEGORICAL) + BINARY_FEATURES.index(row["Feature"]),
                "within_ord": ov,
            })

        fp_frame = (
            pd.DataFrame(fp_rows)
            .sort_values(["group_ord", "within_ord"], ascending=[False, True])
            .reset_index(drop=True)
        )

        err_hi = (fp_frame["hi"] - fp_frame["or"]).where(fp_frame["hi"].notna(), 0).tolist()
        err_lo = (fp_frame["or"] - fp_frame["lo"]).where(fp_frame["lo"].notna(), 0).tolist()

        fig_forest_or = go.Figure(go.Scatter(
            x=fp_frame["or"],
            y=fp_frame["y"],
            mode="markers+text",
            marker=dict(
                color=fp_frame["color"].tolist(),
                size=11,
                symbol=fp_frame["symbol"].tolist(),
            ),
            error_x=dict(
                type="data", symmetric=False,
                array=err_hi, arrayminus=err_lo,
                thickness=2, width=6,
            ),
            text=fp_frame["label"],
            textposition="middle right",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "OR: %{x:.2f}<br>"
                "%{text}<extra></extra>"
            ),
        ))
        fig_forest_or.add_vline(
            x=1.0, line_dash="dash", line_color="gray", line_width=1.5,
            annotation_text="OR = 1.0", annotation_position="top",
        )
        fig_forest_or.update_layout(
            xaxis=dict(title="Odds Ratio (log scale)", type="log"),
            yaxis=dict(title=None),
            height=max(350, len(fp_frame) * 70),
            showlegend=False,
            margin=dict(r=230),
        )
        st.plotly_chart(fig_forest_or, use_container_width=True)
        ref_notes = ", ".join(
            f"**{f}** → *{p2_det_fp[f]['reference_category']}*"
            for f in non_binary_cats
        )
        st.caption(
            "Odds ratios on log scale. Values > 1 indicate increased stroke odds; "
            "values < 1 indicate decreased odds. Error bars = 95% CI. "
            f"◆ = reference category (OR = 1.0 by definition). "
            f"Dummy-coding references: {ref_notes}."
        )
