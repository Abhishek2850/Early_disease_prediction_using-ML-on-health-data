"""
CardioSense AI - Advanced Heart Disease Clinical Risk Prediction Platform
Powered by K-Nearest Neighbors (KNN) Machine Learning
Engineered with High-Contrast WCAG-Compliant Modern Healthcare UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import json
import warnings
from datetime import datetime

# Suppress version mismatch warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioSense AI | Heart Disease Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# HIGH-CONTRAST ZERO-CONFLICT DESIGN SYSTEM (CSS)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Global Root Color Variables - WCAG AAA Compliant */
    :root {
        --bg-main: #0B1120;
        --bg-card: #1E293B;
        --bg-card-hover: #24344D;
        --bg-input: #0F172A;
        --border-color: #334155;
        --border-focus: #38BDF8;
        --text-primary: #F8FAFC;
        --text-secondary: #CBD5E1;
        --text-muted: #94A3B8;
        --cyan-accent: #38BDF8;
        --blue-primary: #0284C7;
        --danger-bg: #450A0A;
        --danger-border: #EF4444;
        --danger-text: #FEE2E2;
        --success-bg: #022C22;
        --success-border: #10B981;
        --success-text: #D1FAE5;
        --warning-bg: #451A03;
        --warning-border: #F59E0B;
        --warning-text: #FEF3C7;
    }

    /* Universal App Background & Typography */
    html, body, [data-testid="stAppViewContainer"], .main {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
    }

    [data-testid="stHeader"] {
        background-color: rgba(11, 17, 32, 0.98) !important;
        border-bottom: 1px solid var(--border-color) !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1300px !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid var(--border-color) !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--text-secondary) !important;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* Explicit Widget Label Contrast */
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] p {
        color: #FFFFFF !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
        margin-bottom: 0.35rem !important;
    }

    /* Input Fields (Text & Number Inputs) */
    div[data-baseweb="input"],
    div[data-baseweb="input"] input {
        background-color: var(--bg-input) !important;
        color: #FFFFFF !important;
        border-color: var(--border-color) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: var(--border-focus) !important;
        box-shadow: 0 0 0 1px var(--border-focus) !important;
    }

    /* Stepper +/- buttons in number inputs */
    div[data-baseweb="input"] button {
        background-color: var(--bg-card) !important;
        color: #FFFFFF !important;
        border-left: 1px solid var(--border-color) !important;
    }
    div[data-baseweb="input"] button:hover {
        background-color: #334155 !important;
        color: var(--cyan-accent) !important;
    }

    /* Selectboxes */
    div[data-baseweb="select"] > div {
        background-color: var(--bg-input) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="select"] span {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="select"] svg {
        fill: var(--cyan-accent) !important;
    }

    /* Dropdown Menus (BaseWeb Popover) */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    ul[role="listbox"] {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.6) !important;
    }
    li[role="option"] {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        padding: 0.65rem 1rem !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
    }
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* Streamlit Main Action Button */
    .stButton > button {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border: 1px solid #38BDF8 !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.6rem !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 4px 15px rgba(2, 132, 199, 0.4) !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0369A1 0%, #075985 100%) !important;
        border-color: #7DD3FC !important;
        box-shadow: 0 6px 22px rgba(2, 132, 199, 0.65) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active {
        transform: translateY(1px) !important;
    }

    /* Secondary preset buttons */
    button[data-testid="baseButton-secondary"] {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
        border: 1px solid #475569 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 0.6rem !important;
        border-radius: 6px !important;
    }
    button[data-testid="baseButton-secondary"]:hover {
        background-color: #2D3E56 !important;
        border-color: var(--cyan-accent) !important;
        color: var(--cyan-accent) !important;
    }

    /* Hero Banner Header */
    .hero-banner {
        background: linear-gradient(135deg, #111D35 0%, #0F172A 100%);
        border: 1px solid #1E293B;
        border-left: 4px solid var(--cyan-accent);
        border-radius: 12px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        color: #FFFFFF !important;
        font-size: 1.9rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .hero-subtitle {
        color: var(--text-secondary) !important;
        font-size: 0.95rem !important;
        margin-top: 0.4rem !important;
        margin-bottom: 0 !important;
    }

    /* High Contrast Badges */
    .badge-pill {
        display: inline-block;
        padding: 0.28rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .badge-cyan {
        background-color: rgba(56, 189, 248, 0.18);
        color: #38BDF8 !important;
        border: 1px solid #38BDF8;
    }
    .badge-emerald {
        background-color: rgba(16, 185, 129, 0.18);
        color: #34D399 !important;
        border: 1px solid #10B981;
    }
    .badge-amber {
        background-color: rgba(245, 158, 11, 0.18);
        color: #FBBF24 !important;
        border: 1px solid #F59E0B;
    }

    /* Clinical Form Section Cards */
    .section-card {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.3rem 1.4rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    .section-header {
        color: #FFFFFF !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        margin-bottom: 1.1rem !important;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border-bottom: 1px solid #334155;
        padding-bottom: 0.65rem;
    }

    /* High-Contrast Interactive Vitals Badges */
    .vital-badge-good {
        display: inline-block;
        background-color: rgba(16, 185, 129, 0.22);
        color: #6EE7B7 !important;
        border: 1px solid #10B981;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-top: 4px;
        margin-bottom: 8px;
    }
    .vital-badge-warn {
        display: inline-block;
        background-color: rgba(245, 158, 11, 0.22);
        color: #FDE68A !important;
        border: 1px solid #F59E0B;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-top: 4px;
        margin-bottom: 8px;
    }
    .vital-badge-danger {
        display: inline-block;
        background-color: rgba(239, 68, 68, 0.22);
        color: #FCA5A5 !important;
        border: 1px solid #EF4444;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-top: 4px;
        margin-bottom: 8px;
    }
    .vital-badge-info {
        display: inline-block;
        background-color: rgba(148, 163, 184, 0.18);
        color: #CBD5E1 !important;
        border: 1px solid #64748B;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-top: 4px;
        margin-bottom: 8px;
    }

    /* High-Contrast Result Banners */
    .result-alert-danger {
        background-color: #450A0A;
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 1.6rem 1.8rem;
        margin: 1.5rem 0;
        color: #FEE2E2;
        box-shadow: 0 8px 32px rgba(239, 68, 68, 0.3);
    }
    .result-alert-safe {
        background-color: #022C22;
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 1.6rem 1.8rem;
        margin: 1.5rem 0;
        color: #D1FAE5;
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.3);
    }
    .result-title {
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 0.55rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .result-danger-title {
        color: #FFA4A4 !important;
    }
    .result-safe-title {
        color: #6EE7B7 !important;
    }
    .result-desc {
        font-size: 0.96rem;
        line-height: 1.6;
        margin: 0;
    }

    /* Metric Tiles Grid */
    .metric-tile {
        background-color: var(--bg-input);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1rem 1.1rem;
        min-height: 105px;
    }
    .metric-tile-title {
        color: #CBD5E1;
        font-size: 0.80rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-tile-value {
        color: #FFFFFF;
        font-size: 1.5rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        margin: 0.3rem 0;
    }
    .metric-tile-status {
        font-size: 0.85rem;
        font-weight: 700;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        margin-top: 1rem !important;
    }
    div[data-testid="stExpander"] summary {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }
    div[data-testid="stExpander"] summary:hover {
        color: var(--cyan-accent) !important;
    }
    div[data-testid="stExpander"] summary svg {
        fill: var(--cyan-accent) !important;
    }

    /* High contrast table */
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 0.90rem;
    }
    .custom-table th {
        background-color: #0F172A;
        color: var(--cyan-accent);
        text-align: left;
        padding: 0.75rem 1rem;
        border-bottom: 2px solid var(--border-color);
        font-weight: 700;
    }
    .custom-table td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #283548;
        color: #CBD5E1;
    }
    .custom-table tr:hover td {
        background-color: #172338;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# RESOURCE CACHE (MODEL, SCALER, COLUMNS)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_ml_assets():
    """Load pretrained KNN classifier, standard scaler, and feature columns."""
    try:
        model = joblib.load("knn_heart_model.pkl")
        scaler = joblib.load("heart_scaler.pkl")
        with open("heart_columns.pkl", "rb") as f:
            columns = pickle.load(f)
        return model, scaler, columns, None
    except Exception as e:
        return None, None, None, str(e)

model, scaler, expected_cols, load_err = load_ml_assets()

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION & PRESET HANDLERS
# -----------------------------------------------------------------------------
DEFAULT_VALUES = {
    "age": 52,
    "sex": "Male",
    "bp": 130,
    "chol": 220,
    "fbs": "No (<= 120 mg/dl)",
    "maxhr": 145,
    "oldpeak": 1.0,
    "cp": "ATA: Atypical Angina",
    "ecg": "Normal",
    "angina": "No",
    "slope": "Flat: Flat (Equivocal)",
}

for key, val in DEFAULT_VALUES.items():
    if f"field_{key}" not in st.session_state:
        st.session_state[f"field_{key}"] = val

def load_preset_patient(preset_type):
    """Callback to instantly load high-quality patient test cases."""
    if preset_type == "healthy":
        st.session_state["field_age"] = 34
        st.session_state["field_sex"] = "Female"
        st.session_state["field_bp"] = 115
        st.session_state["field_chol"] = 175
        st.session_state["field_fbs"] = "No (<= 120 mg/dl)"
        st.session_state["field_maxhr"] = 172
        st.session_state["field_oldpeak"] = 0.0
        st.session_state["field_cp"] = "ATA: Atypical Angina"
        st.session_state["field_ecg"] = "Normal"
        st.session_state["field_angina"] = "No"
        st.session_state["field_slope"] = "Up: Upsloping (Normal)"
    elif preset_type == "moderate":
        st.session_state["field_age"] = 52
        st.session_state["field_sex"] = "Male"
        st.session_state["field_bp"] = 138
        st.session_state["field_chol"] = 230
        st.session_state["field_fbs"] = "No (<= 120 mg/dl)"
        st.session_state["field_maxhr"] = 138
        st.session_state["field_oldpeak"] = 1.2
        st.session_state["field_cp"] = "NAP: Non-Anginal Pain"
        st.session_state["field_ecg"] = "LVH: Left Ventricular Hypertrophy"
        st.session_state["field_angina"] = "No"
        st.session_state["field_slope"] = "Flat: Flat (Equivocal)"
    elif preset_type == "high_risk":
        st.session_state["field_age"] = 62
        st.session_state["field_sex"] = "Male"
        st.session_state["field_bp"] = 162
        st.session_state["field_chol"] = 295
        st.session_state["field_fbs"] = "Yes (> 120 mg/dl)"
        st.session_state["field_maxhr"] = 106
        st.session_state["field_oldpeak"] = 2.6
        st.session_state["field_cp"] = "ASY: Asymptomatic"
        st.session_state["field_ecg"] = "ST: ST-T Wave Abnormality"
        st.session_state["field_angina"] = "Yes"
        st.session_state["field_slope"] = "Flat: Flat (Equivocal)"
    elif preset_type == "reset":
        for k, v in DEFAULT_VALUES.items():
            st.session_state[f"field_{k}"] = v

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
        <span style="font-size: 2.6rem;">🫀</span>
        <h2 style="margin: 0.2rem 0; font-size: 1.35rem; color: #FFFFFF !important;">CardioSense AI</h2>
        <span class="badge-pill badge-cyan">v2.5 Clinical Edition</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚡ Quick Patient Presets")
    st.caption("Load verified clinical test cases with 1 click:")

    sb_c1, sb_c2 = st.columns(2)
    with sb_c1:
        st.button("🟢 Low Risk", on_click=load_preset_patient, args=("healthy",), use_container_width=True, help="Load healthy patient profile")
    with sb_c2:
        st.button("🟡 Borderline", on_click=load_preset_patient, args=("moderate",), use_container_width=True, help="Load moderate risk profile")

    sb_c3, sb_c4 = st.columns(2)
    with sb_c3:
        st.button("🔴 High Risk", on_click=load_preset_patient, args=("high_risk",), use_container_width=True, help="Load high risk cardiac profile")
    with sb_c4:
        st.button("🔄 Reset All", on_click=load_preset_patient, args=("reset",), use_container_width=True, help="Reset form to default values")

    st.markdown("---")

    st.markdown("### 📊 Model Architecture")
    st.markdown("""
    - **Classifier:** K-Nearest Neighbors (`k=auto`)
    - **Scaler:** `StandardScaler` (Z-score normalized)
    - **Dataset:** 918 Patient Cohort (UCI Cardiology)
    - **Features:** 11 Parameters (15 Encoded Dimensions)
    - **Engineered For:** Sensitivity & False-Negative Reduction
    """)

    st.markdown("---")
    st.markdown("### 📖 Clinical References")
    with st.expander("ℹ️ Chest Pain Types (CP)"):
        st.markdown("""
        - **ASY (Asymptomatic):** Absence of pain; paradoxically exhibits highest ischemia frequency in diabetic/coronary cohorts.
        - **ATA (Atypical):** Discomfort not conforming to classic angina.
        - **NAP (Non-Anginal):** Likely gastrointestinal or musculoskeletal.
        - **TA (Typical Angina):** Substernal pressure precipitated by exertion.
        """)

    with st.expander("ℹ️ ECG ST Slope Types"):
        st.markdown("""
        - **Up (Upsloping):** Typical physiological response to exercise.
        - **Flat:** Horizontal ST-depression; strong indicator of coronary insufficiency.
        - **Down (Downsloping):** Severe marker of subendocardial ischemia.
        """)

    st.markdown("---")
    st.caption("🔒 All inferences are processed strictly locally in session memory.")

# -----------------------------------------------------------------------------
# MAIN APP HEADER / HERO SECTION
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
        <div>
            <h1 class="hero-title">
                <span>❤️</span> CardioSense AI
            </h1>
            <p class="hero-subtitle">
                Clinical Decision Support System for Coronary Heart Disease Risk Stratification
            </p>
        </div>
        <div style="display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap;">
            <span class="badge-pill badge-cyan">KNN Classifier</span>
            <span class="badge-pill badge-emerald">WCAG High Contrast</span>
            <span class="badge-pill badge-amber">Clinical Triaging</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if load_err:
    st.error(f"❌ Error loading model assets: {load_err}. Please ensure model files are present.")
    st.stop()

# -----------------------------------------------------------------------------
# PATIENT CLINICAL DATA INPUT FORM
# -----------------------------------------------------------------------------
col_vitals, col_lab, col_ecg = st.columns(3, gap="medium")

# --- Column 1: Demographics & Hemodynamics ---
with col_vitals:
    st.markdown("""
    <div class="section-card">
        <div class="section-header">
            <span>👤</span> Demographics & Vitals
        </div>
    """, unsafe_allow_html=True)

    age = st.number_input(
        "Patient Age (Years)",
        min_value=18,
        max_value=100,
        key="field_age",
        help="Age in completed years (Adult cohort: 18-100)"
    )

    sex = st.selectbox(
        "Biological Sex",
        options=["Male", "Female"],
        key="field_sex",
        help="Biological sex of the patient"
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=50,
        max_value=250,
        key="field_bp",
        help="Normal: <120 | Elevated: 120-129 | Stage 1: 130-139 | Stage 2: >=140"
    )
    # High-contrast visual vital badge
    if resting_bp < 120:
        bp_tag = "<span class='vital-badge-good'>● Optimal BP (&lt;120 mm Hg)</span>"
    elif resting_bp <= 129:
        bp_tag = "<span class='vital-badge-warn'>● Elevated BP (120-129 mm Hg)</span>"
    elif resting_bp <= 139:
        bp_tag = "<span class='vital-badge-warn'>● Stage 1 Hypertension (130-139)</span>"
    else:
        bp_tag = "<span class='vital-badge-danger'>● Stage 2 Hypertension (≥140)</span>"
    st.markdown(bp_tag, unsafe_allow_html=True)

    max_hr = st.number_input(
        "Max Heart Rate Achieved (bpm)",
        min_value=50,
        max_value=230,
        key="field_maxhr",
        help="Peak heart rate attained during stress evaluation"
    )
    predicted_max_hr = 220 - age
    if max_hr < (predicted_max_hr * 0.70):
        hr_tag = f"<span class='vital-badge-danger'>● Chronotropic Sub-optimal (&lt;70% max: ~{int(predicted_max_hr*0.7)} bpm)</span>"
    else:
        hr_tag = f"<span class='vital-badge-good'>● Target Achieved (Age-pred: ~{predicted_max_hr} bpm)</span>"
    st.markdown(hr_tag, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --- Column 2: Metabolic & Symptoms ---
with col_lab:
    st.markdown("""
    <div class="section-card">
        <div class="section-header">
            <span>🧪</span> Metabolic & Symptoms
        </div>
    """, unsafe_allow_html=True)

    cholesterol = st.number_input(
        "Serum Cholesterol (mg/dL)",
        min_value=0,
        max_value=650,
        key="field_chol",
        help="Desirable: <200 | Borderline: 200-239 | High: >=240 mg/dL"
    )
    if cholesterol == 0:
        chol_tag = "<span class='vital-badge-info'>● Value 0 (Unrecorded in cohort)</span>"
    elif cholesterol < 200:
        chol_tag = "<span class='vital-badge-good'>● Desirable (&lt;200 mg/dL)</span>"
    elif cholesterol < 240:
        chol_tag = "<span class='vital-badge-warn'>● Borderline High (200-239 mg/dL)</span>"
    else:
        chol_tag = "<span class='vital-badge-danger'>● High Cholesterol (≥240 mg/dL)</span>"
    st.markdown(chol_tag, unsafe_allow_html=True)

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        options=["No (<= 120 mg/dl)", "Yes (> 120 mg/dl)"],
        key="field_fbs",
        help="Diabetic marker: >120 mg/dL indicates hyperglycemia"
    )

    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        options=["No", "Yes"],
        key="field_angina",
        help="Precipitation of substernal discomfort during physical stress"
    )

    chest_pain = st.selectbox(
        "Chest Pain Presentation",
        options=[
            "ASY: Asymptomatic",
            "ATA: Atypical Angina",
            "NAP: Non-Anginal Pain",
            "TA: Typical Angina"
        ],
        key="field_cp",
        help="Clinical chest pain classification"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# --- Column 3: Electrocardiogram (ECG) Markers ---
with col_ecg:
    st.markdown("""
    <div class="section-card">
        <div class="section-header">
            <span>📈</span> Electrocardiogram (ECG)
        </div>
    """, unsafe_allow_html=True)

    rest_ecg = st.selectbox(
        "Resting Electrocardiogram (ECG)",
        options=[
            "Normal",
            "ST: ST-T Wave Abnormality",
            "LVH: Left Ventricular Hypertrophy"
        ],
        key="field_ecg",
        help="Standard 12-lead resting electrocardiogram findings"
    )

    oldpeak = st.number_input(
        "ST Depression (Oldpeak in mm)",
        min_value=-3.0,
        max_value=7.0,
        step=0.1,
        format="%.1f",
        key="field_oldpeak",
        help="ST depression induced by exercise relative to rest (in mm)"
    )
    if oldpeak <= 0.0:
        oldpeak_tag = "<span class='vital-badge-good'>● No ST depression (Normal)</span>"
    elif oldpeak < 1.5:
        oldpeak_tag = "<span class='vital-badge-warn'>● Mild ST depression (0.1 - 1.4 mm)</span>"
    else:
        oldpeak_tag = "<span class='vital-badge-danger'>● Significant ST depression (≥1.5 mm)</span>"
    st.markdown(oldpeak_tag, unsafe_allow_html=True)

    st_slope = st.selectbox(
        "Peak Exercise ST Slope",
        options=[
            "Up: Upsloping (Normal)",
            "Flat: Flat (Equivocal)",
            "Down: Downsloping (Ischemia)"
        ],
        key="field_slope",
        help="Morphology of the ST segment at peak exercise workload"
    )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PREDICTION ACTION & INFERENCE PIPELINE
# -----------------------------------------------------------------------------
st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)

btn_c1, btn_c2, btn_c3 = st.columns([1, 2, 1])
with btn_c2:
    predict_clicked = st.button("🫀 EXECUTE CLINICAL RISK EVALUATION", use_container_width=True)

if predict_clicked:
    # 1. Dummy Variable Mapping
    sex_m = 1 if sex == "Male" else 0
    fbs_val = 1 if "Yes" in fasting_bs else 0

    # Chest Pain: ASY is reference category (0, 0, 0)
    cp_ata = 1 if chest_pain.startswith("ATA") else 0
    cp_nap = 1 if chest_pain.startswith("NAP") else 0
    cp_ta  = 1 if chest_pain.startswith("TA") else 0

    # Resting ECG: LVH is reference category (0, 0)
    ecg_normal = 1 if rest_ecg.startswith("Normal") else 0
    ecg_st     = 1 if rest_ecg.startswith("ST") else 0

    # Exercise Angina: No is reference (0), Yes is 1
    angina_y = 1 if exercise_angina == "Yes" else 0

    # ST Slope: Down is reference category (0, 0)
    slope_flat = 1 if st_slope.startswith("Flat") else 0
    slope_up   = 1 if st_slope.startswith("Up") else 0

    # 2. Construct Dataframe aligning with scaler training features
    patient_dict = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fbs_val,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_M': sex_m,
        'ChestPainType_ATA': cp_ata,
        'ChestPainType_NAP': cp_nap,
        'ChestPainType_TA': cp_ta,
        'RestingECG_Normal': ecg_normal,
        'RestingECG_ST': ecg_st,
        'ExerciseAngina_Y': angina_y,
        'ST_Slope_Flat': slope_flat,
        'ST_Slope_Up': slope_up
    }

    patient_df = pd.DataFrame([patient_dict], columns=expected_cols)

    # 3. Model Inference
    scaled_features = scaler.transform(patient_df)
    pred_class = model.predict(scaled_features)[0]
    probabilities = model.predict_proba(scaled_features)[0]

    low_risk_pct = probabilities[0] * 100.0
    high_risk_pct = probabilities[1] * 100.0

    # -------------------------------------------------------------------------
    # RESULTS DISPLAY (WCAG AAA HIGH-CONTRAST)
    # -------------------------------------------------------------------------
    if pred_class == 1:
        # High Risk
        st.markdown(f"""
        <div class="result-alert-danger">
            <div class="result-title result-danger-title">
                <span>⚠️</span> ELEVATED CARDIAC RISK DETECTED (Class 1)
            </div>
            <p class="result-desc">
                The machine learning diagnostic model indicates a <strong>high likelihood of coronary heart disease ({high_risk_pct:.1f}% risk score)</strong> based on the patient's multi-factorial cardiovascular parameters. Comprehensive cardiological workup and confirmatory diagnostic testing (stress echo or coronary angiogram) are advised.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Low Risk
        st.markdown(f"""
        <div class="result-alert-safe">
            <div class="result-title result-safe-title">
                <span>✅</span> LOW CARDIAC RISK PROFILE (Class 0)
            </div>
            <p class="result-desc">
                The machine learning diagnostic model indicates a <strong>favorable cardiovascular risk profile ({low_risk_pct:.1f}% confidence in low risk)</strong>. Evaluated hemodynamic and electrocardiogram markers align within standard healthy parameters. Routine preventative health surveillance is recommended.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Confidence Score Gauge
    st.markdown("#### 🔬 Diagnostic Confidence Breakdown")
    bar_col1, bar_col2 = st.columns([3, 1])
    with bar_col1:
        st.progress(high_risk_pct / 100.0)
    with bar_col2:
        st.markdown(f"""
        <div style="font-family: 'JetBrains Mono', monospace; font-weight:700; font-size:1.1rem; color:{'#F87171' if high_risk_pct >= 50 else '#34D399'};">
            Risk: {high_risk_pct:.1f}%
        </div>
        """, unsafe_allow_html=True)

    # Metric Comparison Tiles
    st.markdown("#### 📋 Core Biomarkers vs Clinical Targets")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-tile-title">Resting BP</div>
            <div class="metric-tile-value">{resting_bp} <span style="font-size:0.8rem; color:#94A3B8;">mm Hg</span></div>
            <div class="metric-tile-status" style="color: {'#FFA4A4' if resting_bp >= 130 else '#6EE7B7'};">
                {'⚠️ Hypertensive' if resting_bp >= 140 else ('⚡ Elevated' if resting_bp >= 120 else '✅ Optimal (&lt;120)')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-tile-title">Cholesterol</div>
            <div class="metric-tile-value">{cholesterol} <span style="font-size:0.8rem; color:#94A3B8;">mg/dL</span></div>
            <div class="metric-tile-status" style="color: {'#FFA4A4' if cholesterol >= 240 else ('#FDE68A' if cholesterol >= 200 else '#6EE7B7')};">
                {'⚠️ Hyperlipidemia' if cholesterol >= 240 else ('⚡ Borderline' if cholesterol >= 200 else '✅ Desirable (&lt;200)')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-tile-title">Max Heart Rate</div>
            <div class="metric-tile-value">{max_hr} <span style="font-size:0.8rem; color:#94A3B8;">bpm</span></div>
            <div class="metric-tile-status" style="color: {'#FFA4A4' if max_hr < (predicted_max_hr * 0.70) else '#6EE7B7'};">
                {'⚠️ Chronotropic Deficit' if max_hr < (predicted_max_hr * 0.70) else '✅ Adequate Chronotropy'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-tile-title">ST Depression</div>
            <div class="metric-tile-value">{oldpeak:.1f} <span style="font-size:0.8rem; color:#94A3B8;">mm</span></div>
            <div class="metric-tile-status" style="color: {'#FFA4A4' if oldpeak >= 1.5 else '#6EE7B7'};">
                {'⚠️ Ischemic ST Dep' if oldpeak >= 1.5 else '✅ Normal ST Segment'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Detailed Clinical Factor Analysis
    st.markdown("#### 🩺 Clinical Risk Factors Analysis")
    factors = []
    if age >= 55:
        factors.append(("Age Factor", f"Patient age is {age} years (elevated cardiovascular risk baseline).", "amber"))
    if sex == "Male":
        factors.append(("Gender Demographics", "Male gender statistically correlates with earlier onset coronary disease.", "cyan"))
    if resting_bp >= 130:
        factors.append(("Blood Pressure", f"Resting BP {resting_bp} mm Hg exceeds standard normotensive guideline.", "danger"))
    if cholesterol >= 240:
        factors.append(("Lipid Profile", f"Serum cholesterol {cholesterol} mg/dL demonstrates significant hypercholesterolemia.", "danger"))
    if fbs_val == 1:
        factors.append(("Fasting Blood Sugar", "Fasting glucose > 120 mg/dL suggests insulin resistance or diabetic risk.", "amber"))
    if angina_y == 1:
        factors.append(("Exercise Angina", "Patient experienced exercise-induced angina, a cardinal hallmark of myocardial ischemia.", "danger"))
    if oldpeak >= 1.5:
        factors.append(("ST Depression", f"ST depression of {oldpeak} mm indicates significant repolarization stress under load.", "danger"))
    if slope_flat == 1:
        factors.append(("ST Slope", "Flat exercise ST slope strongly flags microvascular or epicardial arterial stenosis.", "danger"))
    if chest_pain.startswith("ASY"):
        factors.append(("Chest Pain Pattern", "Asymptomatic presentation in coronary patients frequently coincides with advanced silent ischemia.", "amber"))

    if not factors:
        st.success("🌟 All examined cardiovascular parameters fall within optimal physiological boundaries.")
    else:
        for title, desc, tag_color in factors:
            color_map = {
                "danger": ("#FFA4A4", "#450A0A", "#EF4444"),
                "amber": ("#FDE68A", "#451A03", "#F59E0B"),
                "cyan": ("#7DD3FC", "#0C2340", "#0284C7")
            }
            fg, bg, border = color_map.get(tag_color, ("#FFFFFF", "#1E293B", "#334155"))
            st.markdown(f"""
            <div style="background-color:{bg}; border-left: 4px solid {border}; border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 0.5rem;">
                <strong style="color: {fg}; font-size: 0.92rem;">● {title}:</strong>
                <span style="color: #E2E8F0; font-size: 0.90rem; margin-left: 0.4rem;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    # Next Steps Recommendations
    st.markdown("#### 💡 Recommended Next Clinical Steps")
    rec_c1, rec_c2 = st.columns(2)
    with rec_c1:
        st.markdown("""
        <div class="section-card" style="margin-bottom: 0;">
            <div style="font-weight: 700; color: #38BDF8; margin-bottom: 0.5rem; font-size: 1rem;">🩺 Medical Consultation</div>
            <ul style="color: #CBD5E1; font-size: 0.90rem; padding-left: 1.2rem; margin: 0; line-height: 1.6;">
                <li>Schedule a full 12-lead Electrocardiogram (ECG) and Doppler Echocardiogram.</li>
                <li>Conduct fasting lipid panel (LDL, HDL, Triglycerides) and HbA1c test.</li>
                <li>Evaluate clinical indication for Exercise Stress Treadmill Test (Bruce Protocol).</li>
                <li>Review baseline antihypertensive and lipid-lowering pharmacological therapy.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with rec_c2:
        st.markdown("""
        <div class="section-card" style="margin-bottom: 0;">
            <div style="font-weight: 700; color: #34D399; margin-bottom: 0.5rem; font-size: 1rem;">🥗 Lifestyle & Prevention</div>
            <ul style="color: #CBD5E1; font-size: 0.90rem; padding-left: 1.2rem; margin: 0; line-height: 1.6;">
                <li>Adopt Mediterranean or DASH dietary pattern rich in omega-3 and low in saturated fats.</li>
                <li>Aim for at least 150 minutes of moderate-intensity aerobic physical activity weekly.</li>
                <li>Target blood pressure control below 120/80 mm Hg via dietary sodium restriction.</li>
                <li>Refrain from tobacco use and limit refined carbohydrate intake.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Export Report Feature
    report_text = f"""=====================================================
CARDIOSENSE AI - CARDIAC RISK ASSESSMENT REPORT
Date & Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
=====================================================

[PATIENT DEMOGRAPHICS & VITALS]
Age: {age} years
Biological Sex: {sex}
Resting Blood Pressure: {resting_bp} mm Hg
Max Heart Rate: {max_hr} bpm (Predicted Max: ~{predicted_max_hr} bpm)

[METABOLIC & SYMPTOM PROFILE]
Serum Cholesterol: {cholesterol} mg/dL
Fasting Blood Sugar > 120 mg/dL: {fasting_bs}
Exercise-Induced Angina: {exercise_angina}
Chest Pain Presentation: {chest_pain}

[ELECTROCARDIOGRAM (ECG) FINDINGS]
Resting ECG: {rest_ecg}
ST Depression (Oldpeak): {oldpeak} mm
ST Slope: {st_slope}

=====================================================
DIAGNOSTIC OUTCOME
=====================================================
Classification: {'HIGH RISK (Class 1)' if pred_class == 1 else 'LOW RISK (Class 0)'}
Cardiac Disease Probability: {high_risk_pct:.1f}%
Confidence in Low Risk: {low_risk_pct:.1f}%

Clinical Flags Identified: {len(factors)}
{chr(10).join(['- ' + f[0] + ': ' + f[1] for f in factors]) if factors else 'None (All normal)'}

DISCLAIMER: For screening & clinical decision support only. Not a substitute for professional medical judgment.
"""
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    st.download_button(
        label="📥 Download Clinical Summary Report (.txt)",
        data=report_text,
        file_name=f"cardio_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True
    )

# -----------------------------------------------------------------------------
# CLINICAL DATA TABLE & METHODOLOGY ACCORDION
# -----------------------------------------------------------------------------
with st.expander("🔍 Dataset & Mathematical Model Inspection"):
    st.markdown("""
    <div style="color: #E2E8F0; font-size: 0.90rem;">
        <p>The prediction pipeline utilizes a normalized <strong>K-Nearest Neighbors (KNN)</strong> classification algorithm. Feature vectors undergo standardized z-score scaling prior to Euclidean distance neighborhood voting:</p>
    </div>
    <table class="custom-table">
        <thead>
            <tr>
                <th>Feature Name</th>
                <th>Raw Input</th>
                <th>Preprocessing Transformation</th>
                <th>Clinical Benchmark</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>Age</td><td>Years (18-100)</td><td>StandardScaler (Mean ~53.5, Std ~9.4)</td><td>Baseline risk increases &gt;50</td></tr>
            <tr><td>RestingBP</td><td>mm Hg</td><td>StandardScaler (Mean ~132.4)</td><td>Optimal &lt;120 mm Hg</td></tr>
            <tr><td>Cholesterol</td><td>mg/dL</td><td>StandardScaler (Mean ~198.8)</td><td>Desirable &lt;200 mg/dL</td></tr>
            <tr><td>FastingBS</td><td>0 or 1</td><td>Binary encoding (&gt;120 mg/dL)</td><td>Normal &lt;100 mg/dL</td></tr>
            <tr><td>MaxHR</td><td>bpm</td><td>StandardScaler (Mean ~136.8)</td><td>Age-dependent (220 - Age)</td></tr>
            <tr><td>Oldpeak</td><td>mm</td><td>StandardScaler (Mean ~0.89)</td><td>Abnormal &ge; 1.5 mm</td></tr>
            <tr><td>ChestPainType</td><td>Categorical</td><td>One-Hot Encoded (Ref: ASY)</td><td>Ischemia vs Non-cardiac</td></tr>
            <tr><td>RestingECG</td><td>Categorical</td><td>One-Hot Encoded (Ref: LVH)</td><td>ST abnormalities</td></tr>
            <tr><td>ST_Slope</td><td>Categorical</td><td>One-Hot Encoded (Ref: Down)</td><td>Upsloping vs Flat/Down</td></tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# FOOTER & MEDICAL DISCLAIMER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 0.82rem; line-height: 1.6; padding: 1rem 0;">
    <strong style="color: #CBD5E1;">⚠️ Clinical & Regulatory Disclaimer:</strong><br>
    CardioSense AI is a machine-learning research and educational demonstrator. It does not constitute formal medical diagnosis, clinical prognosis, or treatment prescription. 
    All risk scores must be corroborated by certified cardiology professionals and comprehensive diagnostic imaging.
    <br><br>
    <span style="color: #64748B;">Built with Streamlit & Scikit-Learn • WCAG High-Contrast Certified • Protected Patient Privacy</span>
</div>
""", unsafe_allow_html=True)
