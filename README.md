# Churn Predictor
## 📡 Customer Retention Intelligence System  
### Telecom Customer Churn Prediction & Analysis

End-to-end data analytics project built during a **Data Analyst Internship at Elevate Labs**.  
This project predicts customer churn using **XGBoost**, explains predictions using **SHAP**, and deploys a **Streamlit web app** for real-time risk scoring.

---

##  Problem Statement

Telecom companies lose **15–25% of customers annually** to churn.  
Acquiring a new customer costs **5× more** than retaining an existing one.

This project aims to:

- Identify **which customers are likely to churn**
- Understand **why they churn**
- Segment customers into **actionable risk categories**
- Provide a **no-code tool** for instant churn prediction

---

## Dataset

- **Source:** IBM Telco Customer Churn (Kaggle)  
- **Size:** 7,043 customers × 21 features  
- **Target:** `Churn` (Yes / No)

### Feature Categories

| Type | Examples |
|------|--------|
| Demographics | Gender, SeniorCitizen, Partner |
| Services | InternetService, StreamingTV |
| Account Info | Contract, PaymentMethod, MonthlyCharges, Tenure |
| Target | Churn |

---

## Model Performance

| Metric | Score |
|-------|------|
| ROC-AUC | ~0.84 |
| Accuracy | ~0.81 |
| Precision (Churn) | ~0.67 |
| Recall (Churn) | ~0.72 |

---



