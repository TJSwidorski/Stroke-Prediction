"""
model_dashboard.py — Phase 3 model results dashboard

Run with: streamlit run model_dashboard.py
Separate from dashboard.py (Phase 1/2 analysis).
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="Stroke Prediction — Model Results",
    layout="wide",
)

# ── Constants ──────────────────────────────────────────────────────────────────

CONFIGS = [
    {"name": "Shallow Wide",       "safe": "shallow_wide"},
    {"name": "Medium Dropout",     "safe": "medium_dropout"},
    {"name": "Deep Regularized",   "safe": "deep_regularized"},
    {"name": "Attention Weighted", "safe": "attention_weighted"},
]

CONFIG_COLORS = {
    "Shallow Wide":       "#1f77b4",
    "Medium Dropout":     "#ff7f0e",
    "Deep Regularized":   "#2ca02c",
    "Attention Weighted": "#d62728",
}

EXPECTED_FILES = [
    ("LR results",              "data/lr_results.json",             "python train_logistic.py"),
    ("LR coefficients",         "data/lr_coefficients.csv",         "python train_logistic.py"),
    ("Optuna study",            "data/optuna_study.pkl",            "python train_logistic.py"),
    ("MLP comparison",          "data/mlp_comparison.csv",          "python train_mlp.py"),
    ("MLP attention weights",   "data/mlp_attention_weights.json",  "python train_mlp.py"),
    ("SHAP LR summary plot",    "data/shap_lr_summary.png",         "python shap_analysis.py"),
    ("SHAP LR bar plot",        "data/shap_lr_bar.png",             "python shap_analysis.py"),
    ("SHAP MLP summary plot",   "data/shap_mlp_summary.png",        "python shap_analysis.py"),
    ("SHAP MLP bar plot",       "data/shap_mlp_bar.png",            "python shap_analysis.py"),
    ("Feature importance table","data/feature_importance_comparison.csv", "python shap_analysis.py"),
]
for c in CONFIGS:
    EXPECTED_FILES.extend([
        (f"{c['name']} history", f"data/mlp_{c['safe']}_history.json", "python train_mlp.py"),
        (f"{c['name']} results", f"data/mlp_{c['safe']}_results.json", "python train_mlp.py"),
    ])

METRIC_COLS = ["AUC-ROC", "Sensitivity", "Specificity", "Precision", "NPV", "F1"]


# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_data
def _load_json(path: str):
    with open(path) as f:
        return json.load(f)

@st.cache_data
def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def _safe_load_json(path: str):
    return _load_json(path) if os.path.exists(path) else None

def _safe_load_csv(path: str):
    return _load_csv(path) if os.path.exists(path) else None


def _metrics_row(name: str, results: dict, threshold_key: str) -> dict:
    m = results[threshold_key]
    return {
        "Model":       name,
        "AUC-ROC":     m["roc_auc"],
        "Sensitivity": m["sensitivity"],
        "Specificity": m["specificity"],
        "Precision":   m["precision"],
        "NPV":         m["npv"],
        "F1":          m["f1"],
        "Threshold":   m["threshold"],
    }


def _style_best(df: pd.DataFrame, cols: list):
    """Highlight maximum value in each metric column with green bold."""
    def _hl(s):
        return [
            "background-color: #d4edda; font-weight: bold" if v == s.max() else ""
            for v in s
        ]
    return df.style.apply(_hl, subset=cols).format(
        {c: "{:.4f}" for c in cols + ["Threshold"]}
    )


def _confusion_fig(cm: dict, title: str, sensitivity: float, specificity: float):
    """Annotated 2×2 confusion matrix heatmap."""
    z    = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    text = [
        [f"TN<br>{cm['tn']}", f"FP<br>{cm['fp']}"],
        [f"FN<br>{cm['fn']}", f"TP<br>{cm['tp']}"],
    ]
    # Color: correct predictions darker, errors lighter
    colors = [["#aec6cf", "#f4a261"], ["#f4a261", "#a8d5a2"]]
    fig = go.Figure(go.Heatmap(
        z=z,
        x=["Pred: No Stroke", "Pred: Stroke"],
        y=["Actual: No Stroke", "Actual: Stroke"],
        text=text,
        texttemplate="%{text}",
        textfont={"size": 14},
        colorscale=[[0, "#f4a261"], [0.5, "#f4a261"], [1, "#a8d5a2"]],
        showscale=False,
        customdata=colors,
    ))
    fig.update_layout(
        title=f"{title}<br><sub>Sensitivity {sensitivity:.3f} · Specificity {specificity:.3f}</sub>",
        height=280,
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig


def _count_params(config: dict, input_dim: int = 14) -> int:
    params = 0
    prev = input_dim
    if config.get("use_attention"):
        params += prev * prev + prev        # Dense(input_dim, softmax)
    for units in config["layers"]:
        params += prev * units + units      # Dense(units) + bias
        prev = units
    params += prev * 1 + 1                  # Dense(1, sigmoid)
    return params


# ── Sidebar — file status ──────────────────────────────────────────────────────

with st.sidebar:
    st.header("Pipeline status")
    st.caption("Green = file exists · Red = missing")
    last_cmd = None
    for label, path, cmd in EXPECTED_FILES:
        exists = os.path.exists(path)
        icon   = "✅" if exists else "❌"
        if not exists and cmd != last_cmd:
            st.markdown(f"{icon} **{label}**")
            st.code(cmd, language="bash")
            last_cmd = cmd
        else:
            st.markdown(f"{icon} {label}")


# ── Banner ─────────────────────────────────────────────────────────────────────

st.title("Stroke Prediction — Phase 3 Model Results")
st.info(
    "This dashboard reports Phase 3 modeling results. "
    "For Phase 1 descriptive analysis and Phase 2 hypothesis testing, "
    "see the main dashboard (`streamlit run dashboard.py`)."
)

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Model Overview",
    "📈 Training Curves",
    "🔍 Logistic Regression Detail",
    "🧠 Neural Network Detail",
    "🔬 SHAP Interpretability",
    "⚙️ Methodology",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Model Overview
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    with st.expander("ℹ️ How to read this tab"):
        st.markdown(
            "**AUC-ROC** measures rank-order discrimination (1.0 = perfect, 0.5 = random). "
            "**Sensitivity** (recall) is the fraction of true strokes correctly flagged — "
            "the primary clinical objective. **Specificity** is the fraction of non-stroke "
            "patients correctly cleared. **Precision** (PPV) is the fraction of flagged patients "
            "who actually had a stroke. **NPV** is the fraction of cleared patients who truly "
            "did not. **F1** is the harmonic mean of precision and sensitivity. "
            "Green cells mark the best value in each column across all models."
        )

    lr_data      = _safe_load_json("data/lr_results.json")
    mlp_results  = {
        c["name"]: _safe_load_json(f"data/mlp_{c['safe']}_results.json")
        for c in CONFIGS
    }
    mlp_results  = {k: v for k, v in mlp_results.items() if v is not None}

    if lr_data is None and not mlp_results:
        st.warning("No model results found. Run `train_logistic.py` and `train_mlp.py`.")
    else:
        rows_d, rows_o = [], []
        if lr_data:
            rows_d.append(_metrics_row("Logistic Regression", lr_data, "metrics_default"))
            rows_o.append(_metrics_row("Logistic Regression", lr_data, "metrics_optimal"))
        for c in CONFIGS:
            if c["name"] in mlp_results:
                rows_d.append(_metrics_row(c["name"], mlp_results[c["name"]], "metrics_default"))
                rows_o.append(_metrics_row(c["name"], mlp_results[c["name"]], "metrics_optimal"))

        # Best model cards
        if rows_o:
            best_auc  = max(rows_o, key=lambda r: r["AUC-ROC"])
            best_sens = max(rows_o, key=lambda r: r["Sensitivity"])
            c1, c2    = st.columns(2)
            c1.metric("Best AUC-ROC (optimal threshold)",  best_auc["Model"],
                      f"{best_auc['AUC-ROC']:.4f}")
            c2.metric("Best Sensitivity (optimal threshold)", best_sens["Model"],
                      f"{best_sens['Sensitivity']:.4f}")
            st.divider()

        col_order = ["Model"] + METRIC_COLS + ["Threshold"]

        st.subheader("At default threshold (0.50)")
        if rows_d:
            st.dataframe(
                _style_best(pd.DataFrame(rows_d)[col_order], METRIC_COLS),
                use_container_width=True, hide_index=True,
            )

        st.subheader("At optimal threshold (targeting ≥ 85% sensitivity)")
        if rows_o:
            st.dataframe(
                _style_best(pd.DataFrame(rows_o)[col_order], METRIC_COLS),
                use_container_width=True, hide_index=True,
            )

        # Plain-English threshold explanation
        st.divider()
        st.subheader("Threshold optimization explained")
        lines = []
        all_rows_for_text = (
            [("Logistic Regression", lr_data)] if lr_data else []
        ) + [(c["name"], mlp_results[c["name"]]) for c in CONFIGS if c["name"] in mlp_results]

        for name, d in all_rows_for_text:
            opt = d["metrics_optimal"]
            lines.append(
                f"- **{name}**: threshold lowered from 0.50 to **{opt['threshold']:.3f}**, "
                f"achieving sensitivity {opt['sensitivity']:.1%} at the cost of specificity "
                f"{opt['specificity']:.1%} (vs. {d['metrics_default']['specificity']:.1%} at 0.50)."
            )
        st.markdown(
            "Each model's optimal threshold is the *highest* decision boundary on the training "
            "ROC curve that still catches ≥ 85% of true stroke cases. Lower thresholds increase "
            "sensitivity (fewer missed strokes) but decrease specificity (more unnecessary referrals).\n\n"
            + "\n".join(lines)
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Training Curves
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    with st.expander("ℹ️ How to read training curves"):
        st.markdown(
            "**Training vs. validation gap**: a large gap between training and validation curves "
            "means the model is memorizing training data (overfitting) rather than generalizing. "
            "**Flat validation + declining training loss** is the key overfitting signal. "
            "The vertical dashed line marks the epoch where validation AUC was highest — "
            "this is where `restore_best_weights=True` saved the model before early stopping triggered."
        )

    histories = {}
    for c in CONFIGS:
        path = f"data/mlp_{c['safe']}_history.json"
        if os.path.exists(path):
            histories[c["name"]] = _load_json(path)

    if not histories:
        st.warning("No training history files found. Run `train_mlp.py`.")
    else:
        # ── Overview: all configs on same axes ──
        st.subheader("All configurations — AUC-ROC during training")

        fig_auc = go.Figure()
        fig_loss = go.Figure()

        for name, h in histories.items():
            color  = CONFIG_COLORS.get(name, "#666")
            epochs = list(range(1, len(h["loss"]) + 1))

            # Determine key names (keras may suffix with _1 etc.)
            auc_key     = next((k for k in h if k == "auc" or k.startswith("auc")), None)
            val_auc_key = next((k for k in h if k == "val_auc" or k.startswith("val_auc")), None)

            if auc_key:
                fig_auc.add_trace(go.Scatter(
                    x=epochs, y=h[auc_key], name=f"{name} train",
                    line=dict(color=color, width=2),
                ))
            if val_auc_key:
                fig_auc.add_trace(go.Scatter(
                    x=epochs, y=h[val_auc_key], name=f"{name} val",
                    line=dict(color=color, width=2, dash="dash"),
                ))

            fig_loss.add_trace(go.Scatter(
                x=epochs, y=h["loss"], name=f"{name} train",
                line=dict(color=color, width=2),
            ))
            if "val_loss" in h:
                fig_loss.add_trace(go.Scatter(
                    x=epochs, y=h["val_loss"], name=f"{name} val",
                    line=dict(color=color, width=2, dash="dash"),
                ))

        fig_auc.update_layout(
            title="AUC-ROC during training",
            xaxis_title="Epoch", yaxis_title="AUC-ROC",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=380,
        )
        fig_loss.update_layout(
            title="Loss during training",
            xaxis_title="Epoch", yaxis_title="Loss",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=380,
        )
        st.plotly_chart(fig_auc,  use_container_width=True)
        st.plotly_chart(fig_loss, use_container_width=True)

        # ── Single-config detailed view ──
        st.divider()
        st.subheader("Single-config detailed view")
        sel_name = st.selectbox("Select config:", list(histories.keys()), key="tab2_sel")
        h = histories[sel_name]
        color = CONFIG_COLORS.get(sel_name, "#1f77b4")
        epochs = list(range(1, len(h["loss"]) + 1))

        auc_key     = next((k for k in h if k == "auc"     or k.startswith("auc")),     "auc")
        val_auc_key = next((k for k in h if k == "val_auc" or k.startswith("val_auc")), "val_auc")

        best_ep = int(np.argmax(h.get(val_auc_key, [0]))) + 1

        fig_detail = make_subplots(
            rows=1, cols=2,
            subplot_titles=["AUC-ROC", "Loss"],
        )
        for col_idx, (train_key, val_key, ylabel) in enumerate(
            [(auc_key, val_auc_key, "AUC-ROC"), ("loss", "val_loss", "Loss")], start=1
        ):
            if train_key in h:
                fig_detail.add_trace(go.Scatter(
                    x=epochs, y=h[train_key], name="Train",
                    line=dict(color=color, width=2),
                    showlegend=(col_idx == 1),
                ), row=1, col=col_idx)
            if val_key in h:
                fig_detail.add_trace(go.Scatter(
                    x=epochs, y=h[val_key], name="Validation",
                    line=dict(color=color, width=2, dash="dash"),
                    showlegend=(col_idx == 1),
                ), row=1, col=col_idx)
            # Early stopping marker
            fig_detail.add_vline(
                x=best_ep, line_dash="dot", line_color="gray",
                annotation_text=f"Best (ep {best_ep})",
                annotation_position="top right",
                row=1, col=col_idx,
            )

        fig_detail.update_layout(height=380, title_text=f"{sel_name} — training detail")
        st.plotly_chart(fig_detail, use_container_width=True)
        st.caption(
            f"Vertical dotted line = epoch {best_ep} (highest validation AUC). "
            "With `restore_best_weights=True`, the saved model uses weights from this epoch."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Logistic Regression Detail
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    with st.expander("ℹ️ How to read this tab"):
        st.markdown(
            "**Optimization history**: each dot is one Optuna trial; the line shows the best "
            "AUC-ROC found so far. Rapid improvement then plateau means the search converged. "
            "**Coefficients**: positive = feature increases stroke probability; zero = feature "
            "eliminated by L1 regularization. **Odds ratios**: an OR of 2 means the odds of "
            "stroke are twice as high per 1-unit increase in that (scaled) feature, holding "
            "others constant. OR < 1 is protective."
        )

    lr_data  = _safe_load_json("data/lr_results.json")
    coef_df  = _safe_load_csv("data/lr_coefficients.csv")

    # ── Section 1 — Optuna ──
    st.subheader("Hyperparameter optimization (Optuna)")
    if not os.path.exists("data/optuna_study.pkl"):
        st.warning("optuna_study.pkl not found. Run `python train_logistic.py`.")
    else:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = joblib.load("data/optuna_study.pkl")

            completed = [t for t in study.trials if t.state.name == "COMPLETE"]
            trial_df  = pd.DataFrame({
                "Trial":    [t.number for t in completed],
                "AUC_ROC":  [t.value  for t in completed],
                "C":        [t.params["C"]        for t in completed],
                "l1_ratio": [t.params["l1_ratio"] for t in completed],
            })
            trial_df["Best so far"] = trial_df["AUC_ROC"].cummax()

            # Best params as metric cards
            bp = study.best_params
            ca, cb, cc = st.columns(3)
            ca.metric("Best CV AUC-ROC", f"{study.best_value:.4f}")
            cb.metric("Best C",          f"{bp['C']:.5f}")
            cc.metric("Best l1_ratio",   f"{bp['l1_ratio']:.4f}")
            st.divider()

            c_left, c_right = st.columns(2)

            with c_left:
                # Optimization history
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(
                    x=trial_df["Trial"], y=trial_df["AUC_ROC"],
                    mode="markers", name="Trial AUC-ROC",
                    marker=dict(color="#aec6cf", size=5),
                ))
                fig_hist.add_trace(go.Scatter(
                    x=trial_df["Trial"], y=trial_df["Best so far"],
                    mode="lines", name="Best so far",
                    line=dict(color="#1f77b4", width=2),
                ))
                fig_hist.update_layout(
                    title="Optimization history",
                    xaxis_title="Trial", yaxis_title="CV AUC-ROC",
                    height=350,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with c_right:
                # Hyperparameter importance
                try:
                    imp   = optuna.importance.get_param_importances(study)
                    imp_df = pd.DataFrame(
                        list(imp.items()), columns=["Parameter", "Importance"]
                    ).sort_values("Importance")
                    fig_imp = px.bar(
                        imp_df, x="Importance", y="Parameter",
                        orientation="h", title="Hyperparameter importance",
                        height=350,
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                except Exception:
                    st.info("Hyperparameter importance not available for this study.")

            # Scatter: C vs l1_ratio colored by AUC-ROC
            fig_scatter = px.scatter(
                trial_df, x="C", y="l1_ratio", color="AUC_ROC",
                log_x=True,
                color_continuous_scale="Viridis",
                title="Search space — all 100 trials (log C × l1_ratio, colored by AUC-ROC)",
                labels={"AUC_ROC": "CV AUC-ROC"},
                height=380,
            )
            fig_scatter.add_trace(go.Scatter(
                x=[bp["C"]], y=[bp["l1_ratio"]],
                mode="markers",
                marker=dict(symbol="star", size=16, color="red", line=dict(width=1, color="white")),
                name="Best trial",
            ))
            st.plotly_chart(fig_scatter, use_container_width=True)

        except Exception as exc:
            st.error(f"Could not load Optuna study: {exc}")

    # ── Section 2 — Coefficient plot ──
    st.divider()
    st.subheader("ElasticNet coefficients")
    if coef_df is None:
        st.warning("lr_coefficients.csv not found. Run `python train_logistic.py`.")
    else:
        coef_sorted = coef_df.sort_values("coefficient")
        colors = [
            "#c0392b" if c > 0 else ("#2980b9" if c < 0 else "#95a5a6")
            for c in coef_sorted["coefficient"]
        ]
        fig_coef = go.Figure(go.Bar(
            x=coef_sorted["coefficient"],
            y=coef_sorted["feature"],
            orientation="h",
            marker_color=colors,
        ))
        fig_coef.add_vline(x=0, line_color="black", line_width=1)
        fig_coef.update_layout(
            title="ElasticNet coefficients (positive = higher stroke risk)",
            xaxis_title="Coefficient",
            yaxis_title="",
            height=max(350, len(coef_sorted) * 28),
            margin=dict(l=180),
        )
        st.plotly_chart(fig_coef, use_container_width=True)
        zeroed = coef_df[coef_df["coefficient"] == 0.0]["feature"].tolist()
        if zeroed:
            st.caption(f"Features zeroed by L1 (gray bars): {', '.join(zeroed)}")

    # ── Section 3 — Odds ratio plot ──
    st.divider()
    st.subheader("Odds ratios")
    if coef_df is not None:
        or_df = coef_df[coef_df["coefficient"] != 0.0].sort_values("odds_ratio", ascending=True)
        colors_or = ["#c0392b" if v > 1.0 else "#2980b9" for v in or_df["odds_ratio"]]
        fig_or = go.Figure(go.Bar(
            x=or_df["odds_ratio"],
            y=or_df["feature"],
            orientation="h",
            marker_color=colors_or,
        ))
        fig_or.add_vline(x=1.0, line_color="black", line_width=1, line_dash="dash")
        fig_or.update_layout(
            title="Odds ratios — non-zero features (log scale)",
            xaxis_title="Odds ratio", xaxis_type="log",
            yaxis_title="",
            height=max(300, len(or_df) * 28),
            margin=dict(l=180),
        )
        st.plotly_chart(fig_or, use_container_width=True)
        st.caption(
            "Adjusted odds ratios from logistic regression — compare to Phase 2 unadjusted ORs. "
            "Features with coefficient = 0 (zeroed by L1) are excluded. "
            "95% CIs are not shown: ElasticNet regularization biases standard errors, "
            "making Wald-based CIs unreliable for regularized models."
        )

    # ── Section 4 — Confusion matrices ──
    st.divider()
    st.subheader("Confusion matrices")
    if lr_data is None:
        st.warning("lr_results.json not found.")
    else:
        md, mo = lr_data["metrics_default"], lr_data["metrics_optimal"]
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                _confusion_fig(md["confusion_matrix"], "Default threshold (0.50)",
                               md["sensitivity"], md["specificity"]),
                use_container_width=True,
            )
        with col_b:
            st.plotly_chart(
                _confusion_fig(mo["confusion_matrix"],
                               f"Optimal threshold ({mo['threshold']:.3f})",
                               mo["sensitivity"], mo["specificity"]),
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Neural Network Detail
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    with st.expander("ℹ️ How to read this tab"):
        st.markdown(
            "**Architecture table** shows the exact model specification used. "
            "Total parameters is computed from the layer sizes and input dimension. "
            "**Training curves** — a flat or declining validation AUC while training AUC rises "
            "indicates overfitting; early stopping prevents the model from saving those weights. "
            "**Attention weights** (Config D only) are the average softmax activations learned "
            "over all training samples — they reflect which input features the network learned "
            "to amplify before processing, but are not directly equivalent to SHAP values."
        )

    sel_cfg = st.selectbox(
        "Select MLP configuration:",
        [c["name"] for c in CONFIGS],
        key="tab4_sel",
    )
    sel_safe = next(c["safe"] for c in CONFIGS if c["name"] == sel_cfg)
    res_path = f"data/mlp_{sel_safe}_results.json"
    hist_path = f"data/mlp_{sel_safe}_history.json"

    res_data = _safe_load_json(res_path)
    hist_data = _safe_load_json(hist_path)

    if res_data is None:
        st.warning(f"Results for {sel_cfg} not found. Run `python train_mlp.py`.")
    else:
        cfg = res_data["config"]

        # ── Section 1 — Architecture summary ──
        st.subheader("Architecture specification")

        try:
            with open("data/feature_columns.json") as f:
                input_dim = len(json.load(f))
        except Exception:
            input_dim = 14

        total_params = _count_params(cfg, input_dim)

        arch_rows = [
            {"Parameter":   "Layer sizes",       "Value": str(cfg["layers"])},
            {"Parameter":   "Input features",    "Value": str(input_dim)},
            {"Parameter":   "Total parameters",  "Value": f"{total_params:,}"},
            {"Parameter":   "Dropout rate",      "Value": str(cfg["dropout_rate"])},
            {"Parameter":   "L2 regularization", "Value": str(cfg["l2_reg"])},
            {"Parameter":   "Batch normalization","Value": str(cfg["use_batch_norm"])},
            {"Parameter":   "Activation",
             "Value":       "LeakyReLU(0.01)" if cfg["use_leaky_relu"] else "ReLU"},
            {"Parameter":   "Input attention",   "Value": str(cfg["use_attention"])},
            {"Parameter":   "Learning rate",     "Value": str(cfg["learning_rate"])},
            {"Parameter":   "Batch size",        "Value": str(cfg["batch_size"])},
            {"Parameter":   "Optimal threshold", "Value": f"{res_data['optimal_threshold']:.4f}"},
        ]
        st.dataframe(pd.DataFrame(arch_rows), use_container_width=True, hide_index=True)

        # ── Section 2 — Training curves ──
        st.divider()
        st.subheader("Training curves")
        if hist_data is None:
            st.warning(f"History for {sel_cfg} not found.")
        else:
            color  = CONFIG_COLORS.get(sel_cfg, "#1f77b4")
            epochs = list(range(1, len(hist_data["loss"]) + 1))
            auc_key     = next((k for k in hist_data if k == "auc"     or k.startswith("auc")),     "auc")
            val_auc_key = next((k for k in hist_data if k == "val_auc" or k.startswith("val_auc")), "val_auc")
            best_ep = int(np.argmax(hist_data.get(val_auc_key, [0]))) + 1

            fig4 = make_subplots(rows=2, cols=2, subplot_titles=[
                "Training loss", "Validation loss",
                "Training AUC",  "Validation AUC",
            ])
            pairs = [
                ("loss",     1, 1), ("val_loss", 1, 2),
                (auc_key,    2, 1), (val_auc_key, 2, 2),
            ]
            for key, row, col in pairs:
                if key in hist_data:
                    fig4.add_trace(go.Scatter(
                        x=epochs, y=hist_data[key],
                        line=dict(color=color, width=2),
                        showlegend=False,
                    ), row=row, col=col)
                    fig4.add_vline(
                        x=best_ep, line_dash="dot", line_color="gray",
                        row=row, col=col,
                    )
            fig4.update_layout(height=460, title_text=f"{sel_cfg} — detailed training")
            st.plotly_chart(fig4, use_container_width=True)
            st.caption(f"Dotted vertical line = epoch {best_ep} (best validation AUC).")

        # ── Section 3 — Confusion matrices ──
        st.divider()
        st.subheader("Confusion matrices")
        md, mo = res_data["metrics_default"], res_data["metrics_optimal"]
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                _confusion_fig(md["confusion_matrix"], "Default threshold (0.50)",
                               md["sensitivity"], md["specificity"]),
                use_container_width=True,
            )
        with col_b:
            st.plotly_chart(
                _confusion_fig(mo["confusion_matrix"],
                               f"Optimal threshold ({mo['threshold']:.3f})",
                               mo["sensitivity"], mo["specificity"]),
                use_container_width=True,
            )

        # ── Section 4 — Attention weights (Config D only) ──
        if cfg.get("use_attention") and os.path.exists("data/mlp_attention_weights.json"):
            st.divider()
            st.subheader("Input attention weights")
            attn = _load_json("data/mlp_attention_weights.json")
            attn_df = (
                pd.DataFrame(list(attn.items()), columns=["feature", "weight"])
                .sort_values("weight", ascending=True)
            )
            fig_attn = go.Figure(go.Bar(
                x=attn_df["weight"],
                y=attn_df["feature"],
                orientation="h",
                marker_color="#9b59b6",
            ))
            fig_attn.update_layout(
                title="Learned input attention weights",
                xaxis_title="Attention weight",
                height=max(300, len(attn_df) * 28),
                margin=dict(l=180),
            )
            st.plotly_chart(fig_attn, use_container_width=True)
            st.caption(
                "Learned input attention weights — higher values indicate features the network "
                "weighted more heavily before processing. Weights sum to 1.0 (softmax). "
                "These are averaged across all training samples."
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SHAP Interpretability
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    with st.expander("ℹ️ How to read SHAP plots"):
        st.markdown(
            "**Beeswarm (summary) plot**: each dot is one patient. Position on the x-axis shows "
            "the SHAP value — how much that feature pushed the prediction toward stroke (positive) "
            "or away (negative). Color shows the feature's actual value (red = high, blue = low). "
            "**Bar plot**: mean absolute SHAP — the average magnitude of each feature's contribution "
            "regardless of direction; a global importance ranking. "
            "**Comparison table**: all importance methods normalized to [0, 1] within each column. "
            "Features that rank highly across multiple methods have the strongest evidence for "
            "clinical relevance."
        )

    # ── Section 1 — LR SHAP ──
    st.subheader("Logistic Regression SHAP")
    col_lr1, col_lr2 = st.columns(2)
    with col_lr1:
        if os.path.exists("data/shap_lr_summary.png"):
            st.image("data/shap_lr_summary.png", caption="SHAP beeswarm — logistic regression")
        else:
            st.info("shap_lr_summary.png not found. Run `python shap_analysis.py`.")
    with col_lr2:
        if os.path.exists("data/shap_lr_bar.png"):
            st.image("data/shap_lr_bar.png", caption="Mean |SHAP| — logistic regression")
        else:
            st.info("shap_lr_bar.png not found.")

    # ── Section 2 — MLP SHAP ──
    st.divider()
    best_mlp_label = "best MLP config (by AUC-ROC)"
    if os.path.exists("data/mlp_comparison.csv"):
        comp = _load_csv("data/mlp_comparison.csv")
        best_mlp_label = comp.loc[comp["AUC_ROC"].idxmax(), "Config"]

    st.subheader(f"MLP SHAP — {best_mlp_label}")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        if os.path.exists("data/shap_mlp_summary.png"):
            st.image("data/shap_mlp_summary.png", caption=f"SHAP beeswarm — {best_mlp_label}")
        else:
            st.info("shap_mlp_summary.png not found. Run `python shap_analysis.py`.")
    with col_m2:
        if os.path.exists("data/shap_mlp_bar.png"):
            st.image("data/shap_mlp_bar.png", caption=f"Mean |SHAP| — {best_mlp_label}")
        else:
            st.info("shap_mlp_bar.png not found.")

    # ── Section 3 — Comparison table ──
    st.divider()
    st.subheader("Feature importance comparison (all methods)")
    imp_path = "data/feature_importance_comparison.csv"
    if not os.path.exists(imp_path):
        st.info("feature_importance_comparison.csv not found. Run `python shap_analysis.py`.")
    else:
        imp_df = _load_csv(imp_path)
        num_cols = [c for c in ["lr_coef_abs", "shap_lr", "shap_mlp", "attention_weight"]
                    if c in imp_df.columns]

        styled = (
            imp_df.style
            .background_gradient(subset=num_cols, cmap="YlOrRd", axis=0)
            .format({c: "{:.3f}" for c in num_cols})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.caption(
            "All importance scores normalized to 0–1 within each method. "
            "Convergence across methods strengthens confidence in a feature's clinical relevance."
        )

        # Per-feature highlight
        st.divider()
        sel_feat = st.selectbox(
            "Highlight feature across all methods:",
            imp_df["feature"].tolist(),
            key="tab5_feat",
        )
        row = imp_df[imp_df["feature"] == sel_feat].iloc[0]
        cols5 = st.columns(len(num_cols))
        labels_map = {
            "lr_coef_abs":      "LR Coef |β|",
            "shap_lr":          "SHAP (LR)",
            "shap_mlp":         "SHAP (MLP)",
            "attention_weight": "Attention",
        }
        for col_w, col_name in zip(cols5, num_cols):
            val = row[col_name]
            col_w.metric(
                labels_map.get(col_name, col_name),
                f"{val:.3f}" if pd.notna(val) else "N/A",
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Methodology
# ══════════════════════════════════════════════════════════════════════════════

with tab6:
    st.header("Methodology & Design Decisions")

    st.subheader("Data flow")
    st.code(
        "stroke_data_clean.csv\n"
        "    └── preprocessing.py          (feature selection, encoding, scaling)\n"
        "            ├── train_logistic.py  (ElasticNet LR + Optuna)\n"
        "            └── train_mlp.py       (4 MLP configurations)\n"
        "                    └── shap_analysis.py   (SHAP values + comparison)\n"
        "                            └── model_dashboard.py  (this app)",
        language="text",
    )

    st.divider()

    with st.expander("Class imbalance strategy", expanded=True):
        st.markdown(
            "Only **4.9%** of patients had strokes. A naïve model predicting 'no stroke' for "
            "everyone achieves 95% accuracy but **0% sensitivity** — clinically useless. "
            "\n\n"
            "We used **balanced class weights** (`sklearn.utils.class_weight.compute_class_weight`), "
            "which scale each sample's loss contribution so the minority class receives proportionally "
            "more influence during training. The resulting weight ratio is approximately **1:19.5** "
            "(stroke vs. no-stroke)."
            "\n\n"
            "**Why not SMOTE?** Synthetic Minority Oversampling generates new minority-class "
            "samples by interpolating between existing ones. In a medical context this risks "
            "creating clinically unrealistic feature combinations. Class weighting achieves the "
            "same gradient-level effect without synthesizing data, and the test set remains an "
            "unmodified sample of the true distribution."
        )

    with st.expander("ElasticNet regularization"):
        st.markdown(
            "ElasticNet combines L1 (Lasso) and L2 (Ridge) penalties:\n\n"
            "- **L1** drives some coefficients exactly to zero, performing automatic feature "
            "selection — useful when some predictors are redundant.\n"
            "- **L2** shrinks all coefficients smoothly and handles correlated features well "
            "(L1 arbitrarily picks one of a correlated pair).\n\n"
            "The `l1_ratio` parameter controls the mix: 0 = pure Ridge, 1 = pure Lasso. "
            "Tuning it with `C` (inverse regularization strength) jointly allows the optimizer "
            "to find the best trade-off between sparsity and shrinkage."
        )

    with st.expander("Bayesian hyperparameter optimization (Optuna)"):
        st.markdown(
            "**Grid search** evaluates every combination in a predefined grid — expensive and "
            "unable to explore continuous ranges effectively. "
            "**Random search** samples uniformly with no memory between trials. "
            "\n\n"
            "**Optuna** uses a **Tree-structured Parzen Estimator (TPE)**: after each trial, "
            "it fits a probabilistic model of the objective function and proposes new hyperparameters "
            "in regions likely to improve performance. This finds better results with fewer trials "
            "than grid or random search, and handles log-scale parameters (like `C`) naturally. "
            "\n\n"
            "We ran **100 trials** optimizing 5-fold cross-validated AUC-ROC on the training set."
        )

    with st.expander("Threshold optimization and clinical reasoning"):
        st.markdown(
            "The default 0.5 threshold treats false positives (unnecessary referrals) and "
            "false negatives (missed strokes) as equally costly. Clinically, **a missed stroke "
            "is far more serious** than an unnecessary follow-up.\n\n"
            "We select the **highest decision threshold** on the training ROC curve that still "
            "achieves **≥ 85% sensitivity** — meaning at least 85% of true stroke patients are "
            "flagged. 'Highest threshold' means we maximize specificity subject to meeting the "
            "recall floor, minimizing unnecessary referrals while keeping sensitivity high.\n\n"
            "The 85% target is a practical benchmark; in a clinical deployment the threshold "
            "would be set based on institutional risk tolerance and downstream capacity."
        )

    with st.expander("Four MLP configuration hypotheses"):
        st.markdown(
            "Each configuration tests a distinct modeling hypothesis:\n\n"
            "| Config | Hypothesis |\n"
            "|--------|------------|\n"
            "| **Shallow Wide** [128] | A single wide layer is sufficient — the data is "
            "linearly separable or near-linearly separable after feature engineering |\n"
            "| **Medium Dropout** [64, 32] | Moderate depth with dropout improves generalization "
            "over a shallow network on this small dataset |\n"
            "| **Deep Regularized** [128, 64, 32] | Combined dropout + L2 allows a deeper network "
            "to generalize without memorizing the 4,088-sample training set |\n"
            "| **Attention Weighted** [64, 32] | A learned input attention mask that re-weights "
            "features before processing improves both performance and interpretability |\n\n"
            "All configs share the same optimizer (Adam), training protocol (early stopping on "
            "val_auc, patience 15), and threshold optimization procedure."
        )

    with st.expander("Why SHAP for interpretability"):
        st.markdown(
            "SHAP (SHapley Additive exPlanations) is grounded in cooperative game theory: "
            "each feature's contribution to a prediction is its average marginal contribution "
            "across all possible feature coalitions.\n\n"
            "Advantages over alternatives:\n"
            "- **Permutation importance** measures global importance but loses directionality "
            "and cannot explain individual predictions.\n"
            "- **Logistic regression coefficients** are model-specific and don't apply to "
            "the MLP.\n"
            "- SHAP provides **instance-level explanations** (why did *this patient* get a "
            "high risk score?), handles non-linear interactions, and produces the directional "
            "beeswarm plot critical for clinical communication.\n\n"
            "We use `LinearExplainer` for the LR model (exact analytical SHAP) and "
            "`DeepExplainer` / `KernelExplainer` for the MLP."
        )

    st.divider()
    st.subheader("Limitations")
    st.markdown(
        "1. **No prospective validation.** All models were trained and evaluated on a single "
        "retrospective dataset. Performance on a new patient population is unknown.\n\n"
        "2. **SMOTE not used.** See class imbalance section above.\n\n"
        "3. **Dataset size.** With 5,110 patients (≈ 250 stroke cases), deep learning models "
        "typically underfit relative to their potential. The MLP configurations are modest by "
        "design, but their generalizability to larger or different populations is untested.\n\n"
        "4. **Multiple threshold comparisons.** Five models evaluated at two thresholds each "
        "on the same held-out test set constitutes multiple comparisons. The reported test metrics "
        "should be treated as indicative rather than definitive performance estimates.\n\n"
        "5. **Cross-sectional design.** The dataset captures a single point in time per patient. "
        "Stroke is a longitudinal event; temporal features (trend in glucose, hypertension duration) "
        "are not captured."
    )
