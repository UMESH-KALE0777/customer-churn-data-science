import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/raw/customer_churn.csv")

df = load_data()

# Drop ID column if present
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# Encode categorical columns
df_encoded = df.copy()
label_encoders = {}

for col in df_encoded.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# Split features and target
X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

st.subheader("🔍 Enter Customer Details")

# User inputs
user_input = {}

for col in X.columns:
    if col in label_encoders:
        options = label_encoders[col].classes_
        user_input[col] = st.selectbox(col, options)
    else:
        user_input[col] = st.number_input(col, float(X[col].min()), float(X[col].max()))

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Encode user input
for col, le in label_encoders.items():
    input_df[col] = le.transform(input_df[col])

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.error(f"⚠ Customer is likely to CHURN (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ Customer is NOT likely to churn (Confidence: {prob:.2f})")
