import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

st.title("AHD Prediction App (Useful Features)")

st.write("""
This app predicts the likelihood of a patient being diagnosed with Advanced HIV Disease (AHD)
using a subset of features.
Please enter the patient's details below.
""")

# Load the trained pipeline
try:
    pipeline = joblib.load("AHD_useful_features.pkl")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define the useful features used in training
useful_features = [
    "Age at reporting",
    "Sex",
    "Weight_kg",
    "Height_cm",
    "Last WHO Stage",
    "Last_VL_cat",
    "Latest CD4 Result",
    "days_on_art"
]

# Identify which are categorical and numerical based on how they were treated in training
categorical_useful = ['Sex', 'Last WHO Stage', 'Last_VL_cat', 'Latest CD4 Result']
numerical_useful = ['Age at reporting', 'Weight_kg', 'Height_cm', 'days_on_art']

st.sidebar.header("Patient Input Features")

input_data = {}

# Collect input for the specific useful features
for feature in useful_features:
    if feature == "Age at reporting":
        input_data[feature] = st.sidebar.number_input(feature, 0, 100, 30, help="Patient's age at the time of reporting.")
    elif feature == "Sex":
        input_data[feature] = st.sidebar.selectbox(feature, ["M", "F", "Other", "missing"], help="Patient's biological sex.")
    elif feature == "Weight_kg":
         input_data[feature] = st.sidebar.number_input("Weight (kg)", 20.0, 150.0, 60.0, format="%f", help="Patient's weight in kilograms.")
    elif feature == "Height_cm":
         input_data[feature] = st.sidebar.number_input("Height (cm)", 100.0, 200.0, 170.0, format="%f", help="Patient's height in centimeters.")
    elif feature == "Last WHO Stage":
        input_data[feature] = st.sidebar.selectbox(feature, ["1", "2", "3", "4", "missing"], help="Last recorded WHO clinical stage.")
    elif feature == "Last_VL_cat":
        input_data[feature] = st.sidebar.selectbox("Last VL Category", ["suppressed", "unsuppressed", "missing"], help="Categorized viral load result (suppressed/unsuppressed).")
    elif feature == "Latest CD4 Result":
         # Keep as text input as it was treated as categorical in the model due to varied inputs
         input_data[feature] = st.sidebar.text_input(feature, "", help="Latest CD4 count result. Can be numeric or descriptive (e.g., >200).")
    elif feature == "days_on_art":
         input_data[feature] = st.sidebar.number_input(feature, 0, 10000, 365, help="Number of days the patient has been on ART.")
    else:
         # Fallback for any other useful features not explicitly handled above
         input_data[feature] = st.sidebar.text_input(feature, "")


# Convert input data to DataFrame, ensuring all useful_features are present
input_df = pd.DataFrame([input_data])

# Ensure columns are in the order the model expects (based on useful_features list)
input_df = input_df[useful_features]


st.subheader("Prediction Result")

if st.button("Predict"):
    try:
        # Make prediction
        prediction = pipeline.predict(input_df)
        prediction_proba = pipeline.predict_proba(input_df)[:, 1]

        st.write(f"Prediction: {'AHD' if prediction[0] == 1 else 'No AHD'}")
        st.write(f"Probability of AHD: {prediction_proba[0]:.4f}")

        if prediction[0] == 1:
            st.warning("This patient is predicted to have AHD.")
        else:
            st.success("This patient is predicted to not have AHD.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please check the input values and try again.")
        # Optional: print the dataframe structure for debugging
        st.write("Input DataFrame structure for prediction:")
        st.write(input_df.dtypes)
        st.write("Input DataFrame head:")
        st.write(input_df.head())
