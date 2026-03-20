import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analysis_utils import (
    CATEGORICAL, NUMERIC, ALL_FEATURES,
    stroke_rate_by_category, stroke_rate_by_numeric,
    apply_filters, encoded_for_correlation,
)

st.set_page_config(page_title="Stroke Prediction Analysis", layout="wide")
st.title("Stroke Prediction — Data Analysis Dashboard")

DATA_PATH = "data/stroke_data_clean.csv"
STROKE_COLOR = {"No Stroke": "#4C78A8", "Stroke": "#E45756"}


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["stroke_label"] = df["stroke"].map({0: "No Stroke", 1: "Stroke"})
    return df


try:
    df = load_data()
except FileNotFoundError:
    st.error("Data not found. Run `retrieve_data.py` then `feature_engineering.py` first.")
    st.stop()

overall_rate = df["stroke"].mean() * 100

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Distributions",
    "🎯 Individual Stroke Risk",
    "🔗 Joint Stroke Risk",
    "🔥 Correlation Matrix",
])

# ── Tab 1: Distributions ──────────────────────────────────────────────────────
with tab1:
    st.header("Feature Distributions")
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
    feature2 = st.selectbox("Feature", ALL_FEATURES, key="risk_feat")

    if feature2 in CATEGORICAL:
        risk_df = stroke_rate_by_category(df, feature2)
        fig2 = px.bar(
            risk_df, x=feature2, y="stroke_pct",
            text=risk_df["stroke_pct"].apply(lambda x: f"{x:.1f}%"),
            labels={"stroke_pct": "Stroke Rate (%)"},
            color="stroke_pct",
            color_continuous_scale="Reds",
        )
        fig2.add_hline(
            y=overall_rate, line_dash="dash", line_color="gray",
            annotation_text=f"Overall: {overall_rate:.1f}%",
            annotation_position="top left",
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(
            risk_df[[feature2, "count", "stroke_pct"]]
            .rename(columns={"stroke_pct": "Stroke Rate (%)", "count": "Patients"})
            .set_index(feature2),
            use_container_width=True,
        )

    else:
        n_bins = st.slider("Number of bins", 4, 20, 10)
        risk_df = stroke_rate_by_numeric(df, feature2, n_bins)
        fig2 = px.bar(
            risk_df, x="bin_label", y="stroke_pct",
            text=risk_df["stroke_pct"].apply(lambda x: f"{x:.1f}%"),
            labels={"stroke_pct": "Stroke Rate (%)", "bin_label": feature2},
            color="stroke_pct",
            color_continuous_scale="Reds",
        )
        fig2.add_hline(
            y=overall_rate, line_dash="dash", line_color="gray",
            annotation_text=f"Overall: {overall_rate:.1f}%",
            annotation_position="top left",
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

# ── Tab 3: Joint Stroke Risk ──────────────────────────────────────────────────
with tab3:
    st.header("Joint Stroke Risk")
    st.caption("Select features and set filter values to compute the conditional stroke probability.")

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
                    rng = st.slider(feat, lo, hi, (lo, hi), key=f"j_{feat}")
                    filters[feat] = ("range", rng)

        filtered = apply_filters(df, filters)
        n = len(filtered)
        rate = filtered["stroke"].mean() * 100 if n > 0 else 0.0

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Patients matching filters", f"{n:,}")
        m2.metric("Stroke rate", f"{rate:.1f}%")
        m3.metric("vs. overall rate", f"{rate - overall_rate:+.1f}%")

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
    st.caption("Categorical features are ordinally encoded for Pearson correlation.")

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
