import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('models/fraud_detection_random_forest_model.pkl')
    scaler = joblib.load('models/fraud_detection_scaler.pkl')
    return model, scaler

model, scaler = load_model()

# App title
st.title('🚨 Fraud Detection System')
st.markdown('Detect fraudulent transactions in real-time using Machine Learning')

# Sidebar for input
st.sidebar.header('Transaction Details')

# Input features
transaction_type = st.sidebar.selectbox('Transaction Type', 
                                      ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'])
amount = st.sidebar.number_input('Amount ($)', min_value=0.0, value=1000.0)
oldbalanceOrg = st.sidebar.number_input('Sender Old Balance ($)', min_value=0.0, value=5000.0)
oldbalanceDest = st.sidebar.number_input('Recipient Old Balance ($)', min_value=0.0, value=2000.0)
hour = st.sidebar.slider('Hour of Day', 0, 23, 12)

# Calculate derived features
if st.sidebar.button('Analyze Transaction'):
    # Create feature vector
    features = pd.DataFrame({
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'oldbalanceDest': [oldbalanceDest],
        'errorBalanceOrig': [0],  # Simplified for demo
        'errorBalanceDest': [0],
        'zeroBalanceAfterTransferOrig': [1 if oldbalanceOrg - amount <= 0 else 0],
        'origBalanceZero': [1 if oldbalanceOrg == 0 else 0],
        'isDestMerchant': [0],  # Simplified
        'isHighRiskType': [1 if transaction_type in ['TRANSFER', 'CASH_OUT'] else 0],
        'hour': [hour],
        'type_CASH_IN': [1 if transaction_type == 'CASH_IN' else 0],
        'type_CASH_OUT': [1 if transaction_type == 'CASH_OUT' else 0],
        'type_DEBIT': [1 if transaction_type == 'DEBIT' else 0],
        'type_PAYMENT': [1 if transaction_type == 'PAYMENT' else 0],
        'type_TRANSFER': [1 if transaction_type == 'TRANSFER' else 0]
    })
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Prediction", "FRAUD ⚠️" if prediction == 1 else "LEGITIMATE ✅")
    with col2:
        st.metric("Fraud Probability", f"{probability:.2%}")
    
    # Risk assessment
    if probability > 0.8:
        st.error("🚨 HIGH RISK: This transaction shows strong fraud indicators!")
    elif probability > 0.5:
        st.warning("⚠️ MEDIUM RISK: This transaction requires further review.")
    else:
        st.success("✅ LOW RISK: Transaction appears legitimate.")
    
    # Feature importance
    if prediction == 1:
        st.subheader("Why this was flagged:")
        if transaction_type in ['TRANSFER', 'CASH_OUT']:
            st.write("• High-risk transaction type (TRANSFER/CASH_OUT)")
        if oldbalanceOrg - amount <= 0:
            st.write("• Transaction would empty the sender's account")
        if hour in [23, 0, 1, 2, 3, 4]:
            st.write("• Unusual transaction time")

# Model info
with st.expander("ℹ️ About this Model"):
    st.write("""
    **Model Performance:**
    - Accuracy: 99.99%
    - Precision: 100%
    - Recall: 95.24%
    - F1-Score: 97.56%
    
    **Training Data:**
    - 69,857 transactions analyzed
    - Advanced feature engineering
    - SMOTE for handling imbalanced data
    """)