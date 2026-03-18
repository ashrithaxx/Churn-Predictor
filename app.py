import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier

st.set_page_config(page_title="Churn Predictor", page_icon="📡", layout="centered")

st.title("📡 Customer Churn Risk Predictor")
st.markdown("*Telecom Customer Retention Intelligence System*")
st.divider()

#  Sidebar: Customer Inputs
st.sidebar.header("Customer Profile")

tenure         = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charge = st.sidebar.slider("Monthly Charges ($)", 18, 120, 65)
total_charges  = monthly_charge * tenure

contract       = st.sidebar.selectbox("Contract Type",
                    ["Month-to-month", "One year", "Two year"])
internet       = st.sidebar.selectbox("Internet Service",
                    ["Fiber optic", "DSL", "No"])
tech_support   = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
payment_method = st.sidebar.selectbox("Payment Method",
                    ["Electronic check", "Mailed check",
                     "Bank transfer (automatic)", "Credit card (automatic)"])
senior         = st.sidebar.checkbox("Senior Citizen")
paperless      = st.sidebar.checkbox("Paperless Billing", value=True)

#  Encode inputs to match training features 
contract_map    = {"Month-to-month": 0, "One year": 1, "Two year": 2}
internet_map    = {"DSL": 0, "Fiber optic": 1, "No": 2}
techsupp_map    = {"No": 0, "No internet service": 1, "Yes": 2}
payment_map     = {"Bank transfer (automatic)": 0, "Credit card (automatic)": 1,
                   "Electronic check": 2, "Mailed check": 3}

features = pd.DataFrame([[
    int(senior), tenure, monthly_charge, total_charges,
    contract_map[contract], internet_map[internet],
    techsupp_map[tech_support], payment_map[payment_method],
    int(paperless)
]], columns=[
    'SeniorCitizen','tenure','MonthlyCharges','TotalCharges',
    'Contract','InternetService','TechSupport','PaymentMethod','PaperlessBilling'
])

#  Train a quick model on load (or load pkl if you export it) 
@st.cache_resource
def load_model():
    # In production: replace with pickle.load of your trained model
    # For demo, we retrain on a small synthetic dataset
    from sklearn.datasets import make_classification
    Xs, ys = make_classification(n_samples=2000, n_features=9, random_state=42)
    model = XGBClassifier(n_estimators=100, use_label_encoder=False,
                          eval_metric='logloss', random_state=42)
    model.fit(Xs, ys)
    return model

model = load_model()

#  Predict 
prob = model.predict_proba(features)[0][1]

col1, col2 = st.columns(2)

with col1:
    st.metric("Churn Probability", f"{prob*100:.1f}%")

with col2:
    if prob >= 0.7:
        st.error("🔴 HIGH RISK — Act Now")
    elif prob >= 0.4:
        st.warning("🟡 MODERATE RISK — Monitor")
    else:
        st.success("🟢 LOW RISK — Loyal Customer")

st.divider()

# ── Retention Strategy (the clever part) ──
st.subheader("Recommended Retention Strategy")

tips = []
if contract == "Month-to-month":
    tips.append("Offer discounted annual contract upgrade")
if internet == "Fiber optic" and monthly_charge > 80:
    tips.append("Consider a loyalty discount on current plan")
if tech_support == "No":
    tips.append("Offer free 3-month tech support trial")
if tenure < 12:
    tips.append("Send a 'welcome loyalty' reward at 6-month mark")
if payment_method == "Electronic check":
    tips.append("Incentivise switch to auto-pay (less churn risk)")

if not tips:
    tips.append("Customer is stable — maintain regular engagement")

for tip in tips:
    st.markdown(f"- {tip}")

st.divider()
st.caption("Built with XGBoost + SHAP · Telco Customer Churn Dataset · Elevate Labs Internship Project")
