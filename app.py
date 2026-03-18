import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Churn Predictor", page_icon="📡", layout="centered")

# Load real trained model
@st.cache_resource
def load_model():
    with open('xgb_churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_columns.pkl', 'rb') as f:
        columns = pickle.load(f)
    return model, columns

model, feature_columns = load_model()

st.title("📡 Customer Churn Risk Predictor")
st.markdown("*Trained on IBM Telco Dataset · XGBoost Model*")
st.divider()

# Sidebar inputs
st.sidebar.header("Enter Customer Details")

tenure         = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charge = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
total_charges  = monthly_charge * tenure 

senior         = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner        = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents     = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
phone_service  = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines",
                    ["No", "Yes", "No phone service"])
internet       = st.sidebar.selectbox("Internet Service",
                    ["Fiber optic", "DSL", "No"])
online_sec     = st.sidebar.selectbox("Online Security",
                    ["No", "Yes", "No internet service"])
online_backup  = st.sidebar.selectbox("Online Backup",
                    ["No", "Yes", "No internet service"])
device_prot    = st.sidebar.selectbox("Device Protection",
                    ["No", "Yes", "No internet service"])
tech_support   = st.sidebar.selectbox("Tech Support",
                    ["No", "Yes", "No internet service"])
streaming_tv   = st.sidebar.selectbox("Streaming TV",
                    ["No", "Yes", "No internet service"])
streaming_mov  = st.sidebar.selectbox("Streaming Movies",
                    ["No", "Yes", "No internet service"])
contract       = st.sidebar.selectbox("Contract Type",
                    ["Month-to-month", "One year", "Two year"])
paperless      = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method",
                    ["Electronic check", "Mailed check",
                     "Bank transfer (automatic)", "Credit card (automatic)"])
gender         = st.sidebar.selectbox("Gender", ["Male", "Female"])
                                      
# LabelEncoder sorts alphabetically, so:
encode = {
    'gender':          {'Female': 0, 'Male': 1},
    'Partner':         {'No': 0, 'Yes': 1},
    'Dependents':      {'No': 0, 'Yes': 1},
    'PhoneService':    {'No': 0, 'Yes': 1},
    'MultipleLines':   {'No': 0, 'No phone service': 1, 'Yes': 2},
    'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
    'OnlineSecurity':  {'No': 0, 'No internet service': 1, 'Yes': 2},
    'OnlineBackup':    {'No': 0, 'No internet service': 1, 'Yes': 2},
    'DeviceProtection':{'No': 0, 'No internet service': 1, 'Yes': 2},
    'TechSupport':     {'No': 0, 'No internet service': 1, 'Yes': 2},
    'StreamingTV':     {'No': 0, 'No internet service': 1, 'Yes': 2},
    'StreamingMovies': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'Contract':        {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    'PaperlessBilling':{'No': 0, 'Yes': 1},
    'PaymentMethod':   {'Bank transfer (automatic)': 0,
                        'Credit card (automatic)': 1,
                        'Electronic check': 2,
                        'Mailed check': 3},
}

row = {
    'gender':           encode['gender'][gender],
    'SeniorCitizen':    1 if senior == "Yes" else 0,
    'Partner':          encode['Partner'][partner],
    'Dependents':       encode['Dependents'][dependents],
    'tenure':           tenure,
    'PhoneService':     encode['PhoneService'][phone_service],
    'MultipleLines':    encode['MultipleLines'][multiple_lines],
    'InternetService':  encode['InternetService'][internet],
    'OnlineSecurity':   encode['OnlineSecurity'][online_sec],
    'OnlineBackup':     encode['OnlineBackup'][online_backup],
    'DeviceProtection': encode['DeviceProtection'][device_prot],
    'TechSupport':      encode['TechSupport'][tech_support],
    'StreamingTV':      encode['StreamingTV'][streaming_tv],
    'StreamingMovies':  encode['StreamingMovies'][streaming_mov],
    'Contract':         encode['Contract'][contract],
    'PaperlessBilling': encode['PaperlessBilling'][paperless],
    'PaymentMethod':    encode['PaymentMethod'][payment_method],
    'MonthlyCharges':   monthly_charge,
    'TotalCharges':     total_charges,
}

input_df = pd.DataFrame([row])[feature_columns]

#  Predict 
prob  = model.predict_proba(input_df)[0][1]
pred  = model.predict(input_df)[0]

# Results 
col1, col2 = st.columns(2)
with col1:
    st.metric("Churn Probability", f"{prob*100:.1f}%")
with col2:
    if prob >= 0.7:
        st.error("🔴 HIGH RISK")
    elif prob >= 0.4:
        st.warning("🟡 MODERATE RISK")
    else:
        st.success("🟢 LOW RISK")

#  Risk bar
st.progress(float(prob))
st.divider()

# Retention Strategy
st.subheader("Recommended Retention Strategy")
tips = []
if contract == "Month-to-month":
    tips.append("Offer a discounted annual or 2-year contract upgrade")
if internet == "Fiber optic" and monthly_charge > 80:
    tips.append("Apply a loyalty discount — high-value customer at risk")
if tech_support == "No" and internet != "No":
    tips.append("Offer a free 3-month tech support trial")
if tenure < 12:
    tips.append("Send a welcome loyalty reward at the 6-month mark")
if payment_method == "Electronic check":
    tips.append("Incentivise switch to auto-pay — reduces churn likelihood")
if online_sec == "No" and internet != "No":
    tips.append("Bundle online security — adds stickiness to the plan")
if not tips:
    tips.append("Customer is stable — maintain regular engagement touchpoints")

for tip in tips:
    st.markdown(f"- {tip}")

st.divider()
st.caption("XGBoost · IBM Telco Dataset · Elevate Labs Internship · Built with Streamlit")
