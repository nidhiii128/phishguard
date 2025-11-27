import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Import feature extraction from the same folder
from .feature_engineering import create_feature_dataframe

@st.cache_resource
def load_model():
    try:
        model_data = joblib.load('models/trained_model.joblib')
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Resolve path to trained model
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/trained_model.joblib"))

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"❌ Model file not found at {MODEL_PATH}. "
        "Please run the training script first."
    )

# Load the saved model
model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
scaler = model_data["scaler"]
feature_columns = model_data["feature_columns"]
needs_scaling = model_data["needs_scaling"]

def predict_url(url: str):
    """
    Input: URL string
    Output: label ("Phishing"/"Legitimate") and confidence in %
    """
    # Extract features for a single URL
    feature_df = create_feature_dataframe([url], [0])  # dummy label
    X = feature_df[feature_columns].fillna(0)

    # Scale if needed
    if needs_scaling:
        X = scaler.transform(X)

    # Predict
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = np.max(proba) * 100

    label = "Phishing" if prediction == 1 else "Legitimate"
    return label, confidence

# Test if running standalone
if __name__ == "__main__":
    test_url = "https://paypal-security-login-check.com"
    lbl, conf = predict_url(test_url)
    print(f"URL: {test_url}\nPrediction: {lbl}\nConfidence: {conf:.2f}%")
