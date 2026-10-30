import streamlit as st
import pandas as pd
from joblib import load

# Load the trained XGBoost model and feature names
model = load('fraud_model.joblib')
feature_names = load('feature_names.joblib')

# Create a Streamlit app
st.title("Fraud Detection App")

# Input fields for transaction details
st.header("Enter Transaction Details")

step           = st.number_input("Step (Hour of Simulation)", min_value=1, max_value=744, value=1)
tx_type        = st.selectbox("Transaction Type", ('CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'))
amount         = st.number_input("Amount (₹)", min_value=0.0, value=10000.0, step=100.0)

st.subheader("Sender Balance")
old_bal_org  = st.number_input("Old Balance (Sender)", min_value=0.0, value=50000.0, step=100.0)
new_bal_orig = st.number_input("New Balance (Sender)", min_value=0.0, value=40000.0, step=100.0)

st.subheader("Receiver Balance")
old_bal_dest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=5000.0, step=100.0)
new_bal_dest = st.number_input("New Balance (Receiver)", min_value=0.0, value=15000.0, step=100.0)

# Build input row matching training features
def build_input(step, tx_type, amount, old_bal_org, new_bal_orig, old_bal_dest, new_bal_dest):
    row = {f: 0 for f in feature_names}
    row['step']              = step
    row['amount']            = amount
    row['oldbalanceOrg']     = old_bal_org
    row['newbalanceOrig']    = new_bal_orig
    row['oldbalanceDest']    = old_bal_dest
    row['newbalanceDest']    = new_bal_dest
    row['balance_diff_orig'] = old_bal_org - new_bal_orig
    row['balance_diff_dest'] = new_bal_dest - old_bal_dest
    type_col = f'type_{tx_type}'
    if type_col in row:
        row[type_col] = 1
    return pd.DataFrame([row])[feature_names]

# Predict on button click
if st.button("Check Transaction"):
    input_df    = build_input(step, tx_type, amount, old_bal_org, new_bal_orig, old_bal_dest, new_bal_dest)
    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.header("Prediction Result")
    if prediction == 1:
        st.error("🚨 This transaction is likely FRAUDULENT.")
    else:
        st.success("✅ This transaction appears LEGITIMATE.")

    st.write(f"Fraud Probability: **{probability * 100:.2f}%**")