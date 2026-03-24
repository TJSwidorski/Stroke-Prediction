import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from analysis_utils import (
    CATEGORICAL, NUMERIC, ALL_FEATURES,
    DEFAULT_BIN_WIDTHS, BIN_WIDTH_OPTIONS,
    feature_stroke_stats, all_features_stats,
    apply_filters, encoded_for_correlation,
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


try:
    df = load_data()
except FileNotFoundError:
    st.error("Data not found. Run `retrieve_data.py` then `feature_engineering.py` first.")
    st.stop()

p0 = df["stroke"].mean()
overall_rate = p0 * 100

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏥 Cohort Overview",
    "📊 Distributions",
    "🎯 Individual Stroke Risk",
    "🔗 Joint Stroke Risk",
    "🔥 Correlation Matrix",
    "🧪 Hypothesis Testing",
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


# ── Tab 1: Distributions ──────────────────────────────────────────────────────
with tab1:
    st.header("Feature Distributions")

    with st.expander("ℹ️ How to read this chart"):
        st.markdown(
            "Shows the **count of patients** for each value of the selected feature. "
            "Use this to understand how the dataset is distributed — e.g., how many patients "
            "are in each age range or work category.\n\n"
            "- **Split by stroke outcome** separates each bar into stroke vs. no-stroke patients, "
            "making it easy to see whether one group is over-represented in a category.\n"
            "- This shows *raw counts*, not rates. A large stroke bar doesn't necessarily mean "
            "high risk — it may simply reflect a large group. See **Individual Stroke Risk** for rates."
        )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        feature = st.selectbox("Feature", ALL_FEATURES, key="dist_feat")
    with col_b:
        split = st.checkbox("Split by stroke outcome", value=True)

    if feature in CATEGORICAL:
        fig = px.histogram(
            df, x=feature,
            color="stroke_label" if split else None,
            barmode="group",
            color_discrete_map=STROKE_COLOR,
            labels={"stroke_label": "Outcome"},
            category_orders={"stroke_label": ["No Stroke", "Stroke"]},
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
