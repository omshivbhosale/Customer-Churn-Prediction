from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------- Part A: load the model ----------
ROOT = Path(__file__).resolve().parent.parent
SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def add_features(df):
    # Must match the feature engineering in 03_modeling.ipynb
    df = df.copy()
    df["num_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    return df


@st.cache_resource
def load_model():
    saved = joblib.load(ROOT / "models" / "churn_model.joblib")
    pipe = saved["model"]

    # The "average customer", used as the starting point for explanations
    data = pd.read_csv(ROOT / "data" / "churn_clean.csv")
    X_all = add_features(data.drop(columns=["Churn"]))
    average_customer = pipe.named_steps["prep"].transform(X_all).mean(axis=0)
    return pipe, saved["threshold"], average_customer


pipe, threshold, average_customer = load_model()

st.set_page_config(page_title="Churn Risk Predictor", page_icon="📉")
st.title("📉 Customer Churn Risk Predictor")
st.write("Enter a customer's details to see how likely they are to leave, and why.")

# ---------- Part B: inputs ----------
st.subheader("Account")
col1, col2 = st.columns(2)
with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 3)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                              "Bank transfer (automatic)",
                                              "Credit card (automatic)"])
with col2:
    monthly = st.number_input("Monthly charges ($)", 18.0, 120.0, 85.0, step=5.0)
    paperless = st.radio("Paperless billing", ["Yes", "No"], horizontal=True)

st.subheader("Customer")
col1, col2 = st.columns(2)
with col1:
    gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
    senior = st.radio("Senior citizen", ["No", "Yes"], horizontal=True)
with col2:
    partner = st.radio("Partner", ["No", "Yes"], horizontal=True)
    dependents = st.radio("Dependents", ["No", "Yes"], horizontal=True)

st.subheader("Services")
col1, col2 = st.columns(2)
with col1:
    phone = st.radio("Phone service", ["Yes", "No"], horizontal=True)
    multiple = st.radio("Multiple lines", ["No", "Yes"], horizontal=True)
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
with col2:
    extras = st.multiselect(
        "Internet add-ons",
        ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
         "StreamingTV", "StreamingMovies"],
    )

# Build one row in exactly the same format as the training data
customer = {
    "gender": gender, "SeniorCitizen": senior, "Partner": partner,
    "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
    "MultipleLines": multiple if phone == "Yes" else "No phone service",
    "InternetService": internet, "Contract": contract,
    "PaperlessBilling": paperless, "PaymentMethod": payment,
    "MonthlyCharges": monthly, "TotalCharges": tenure * monthly,
}
for addon in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
              "StreamingTV", "StreamingMovies"]:
    if internet == "No":
        customer[addon] = "No internet service"
    else:
        customer[addon] = "Yes" if addon in extras else "No"

X_new = add_features(pd.DataFrame([customer]))

# ---------- Part C: predict ----------
proba = pipe.predict_proba(X_new)[0, 1]

st.divider()
st.metric("Churn probability", f"{proba:.0%}")
if proba >= threshold:
    st.error(f"⚠️ High risk (above the {threshold:.0%} threshold): send a retention offer.")
else:
    st.success(f"✅ Low risk (below the {threshold:.0%} threshold): no action needed.")


# ---------- Part D: explain ----------
prep = pipe.named_steps["prep"]
logreg = pipe.named_steps["model"]
names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
x_ready = prep.transform(X_new)[0]
contributions = pd.Series(logreg.coef_[0] * (x_ready - average_customer), index=names)

# Add the one-hot pieces back together: Contract_One year + Contract_Two year + ... -> Contract
by_feature = {}
for col in X_new.columns:
    by_feature[col] = contributions[[n for n in names
                                     if n == col or n.startswith(col + "_")]].sum()
by_feature = pd.Series(by_feature)
top = by_feature.reindex(by_feature.abs().sort_values(ascending=False).index).head(6)

st.subheader("Why?")
st.caption("Red pushes toward churn, green toward staying (compared with an average customer).")
st.bar_chart(pd.DataFrame({"Pushes to churn": top.clip(lower=0),
                           "Pushes to stay": top.clip(upper=0)}),
             horizontal=True, sort=False, color=["#d62728", "#2ca02c"])