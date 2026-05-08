import streamlit as st

st.set_page_config(
    page_title="Stroke Prediction",
    page_icon="🏥",
    layout="wide",
)

import dashboard
import model_dashboard

SECTIONS = [
    "🔬 Phase 1/2 — Data Analysis",
    "🤖 Phase 3 — Model Results",
]

with st.sidebar:
    st.title("Stroke Prediction")
    st.caption("A clinical ML research project")
    st.divider()
    section = st.radio(
        "Navigate to:",
        SECTIONS,
    )
    st.divider()

if section == SECTIONS[0]:
    dashboard.render()
else:
    model_dashboard.render()
