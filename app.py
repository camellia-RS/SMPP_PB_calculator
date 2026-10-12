# ============================================================
# PB Risk Calculator - Streamlit version (no sklearn dependency)
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier

st.set_page_config(page_title="PB Risk Calculator", page_icon="🩺", layout="centered")

st.title("PB Risk Calculator for SMPP")
st.markdown(
    "Predict the probability of **plastic bronchitis (PB)** in children "
    "with severe *Mycoplasma pneumoniae* pneumonia (SMPP)."
)

# ---------- 加载资源 ----------
@st.cache_resource
def load_resources():
    medians = pd.read_csv('imputer_medians.csv', index_col=0).iloc[:, 0]
    features = joblib.load('feature_names.pkl')
    model = CatBoostClassifier()
    model.load_model('catboost_model.cbm')
    return medians, features, model

medians, FEATURES, model = load_resources()

# ---------- 输入 ----------
st.subheader("Patient inputs")
col1, col2 = st.columns(2)

with col1:
    pulmonary_consolidation = st.selectbox("Pulmonary Consolidation", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    single_lobe_consolidation = st.selectbox("Single Lobe Consolidation >2/3", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    atelectasis = st.selectbox("Atelectasis", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    breath_sounds_decreased = st.selectbox("Decreased or Absent Breath Sounds", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    age_group = st.selectbox("Age group", [0, 1], format_func=lambda x: "1 month-3 years" if x == 0 else "3-16 years")
    peak_temperature = st.number_input("Peak Temperature (°C)", min_value=34.0, max_value=43.0, value=38.5, step=0.1)
    ne_pct = st.number_input("Neutrophil Percentage (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)

with col2:
    wbc = st.number_input("WBC (x10^9/L)", min_value=0.1, max_value=100.0, value=10.0, step=0.1)
    ldh = st.number_input("LDH (U/L)", min_value=10.0, max_value=5000.0, value=300.0, step=1.0)
    esr = st.number_input("ESR (mm/h)", min_value=1.0, max_value=200.0, value=30.0, step=1.0)
    crp = st.number_input("CRP (mg/L)", min_value=0.1, max_value=500.0, value=20.0, step=0.1)
    igg = st.number_input("IgG (g/L)", min_value=0.1, max_value=100.0, value=10.0, step=0.1)
    cl = st.number_input("Chloride (mmol/L)", min_value=70.0, max_value=200.0, value=105.0, step=0.1)
    na = st.number_input("Sodium (mmol/L)", min_value=100.0, max_value=180.0, value=138.0, step=0.1)

st.markdown("---")

if st.button("Predict PB probability", type="primary"):
    input_dict = {
        'Pulmonary_Consolidation': pulmonary_consolidation,
        'Single_Lobe_Consolidation': single_lobe_consolidation,
        'Atelectasis': atelectasis,
        'Breath_Sounds_Decreased': breath_sounds_decreased,
        'Peak_Temperature': peak_temperature,
        'Age_group': age_group,
        'NE_Pct': ne_pct,
        'WBC': wbc,
        'LDH': ldh,
        'ESR': esr,
        'CRP': crp,
        'IgG': igg,
        'Cl': cl,
        'Na': na,
    }
    # 按训练时的顺序构造 DataFrame
    row = pd.DataFrame([[input_dict[f] for f in FEATURES]], columns=FEATURES)
    # 手动用中位数填充（虽然所有输入都有值，这里做保险）
    row = row.fillna(medians)

    prob = model.predict_proba(row)[0, 1]

    st.markdown("### Prediction result")
    st.metric(label="Predicted probability of PB", value=f"{prob:.2%}")

    if prob < 0.15:
        st.success(f"**Low risk** (probability = {prob:.1%}). Routine outpatient monitoring; bronchoscopy generally not indicated.")
    elif prob < 0.40:
        st.warning(f"**Intermediate risk** (probability = {prob:.1%}). Close inpatient observation; consider bronchoscopy if clinically indicated.")
    else:
        st.error(f"**High risk** (probability = {prob:.1%}). Early bronchoscopy and intensive management recommended.")

    st.progress(float(prob))

st.markdown("---")
st.caption("This calculator is intended for research and educational use only. It does not replace clinical judgment.")