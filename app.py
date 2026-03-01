import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

@st.cache_resource
def load_artifacts():
    path = Path("output_artifacts")
    try:
        model = pickle.load(open(path / "model.pkl", "rb"))
        scaler = pickle.load(open(path / "scaler.pkl", "rb"))
        feature_columns = pickle.load(open(path / "feature_columns.pkl", "rb"))
        return model, scaler, feature_columns
    except:
        st.error("Model files missing in output_artifacts folder.")
        return None, None, None

model, scaler, feature_columns = load_artifacts()

if feature_columns:

    # Extract encoded categories
    locations = ["other"] + sorted([c.replace("location_", "") for c in feature_columns if c.startswith("location_")])
    area_types = ["Built-up  Area"] + sorted([c.replace("area_type_", "") for c in feature_columns if c.startswith("area_type_")])
    availabilities = ["Ready To Move"] + sorted([c.replace("availability_", "") for c in feature_columns if c.startswith("availability_")])

    st.title("Bengaluru House Price Predictor")
    st.caption("Enter the property details to get an estimated price")

    col1, col2 = st.columns(2)

    with col1:
        total_sqft = st.number_input("Total Area (sq ft)", min_value=100.0, value=1200.0, step=50.0)
        bhk = st.number_input("Bedrooms (BHK)", min_value=1, max_value=10, value=2)
        bath = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
        if bath > bhk + 2:
            st.warning("Number of bathrooms seems unusually high.")
        balcony = st.slider("Balcony", 0, 3, 1)

    with col2:
        location = st.selectbox("Location", locations)
        area_type = st.selectbox("Area Type", area_types)
        availability = st.selectbox("Availability", availabilities)

    if st.button("Predict Price", type="primary"):
        row = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

        row["total_sqft"] = total_sqft
        row["bhk"] = bhk
        row["bath"] = bath
        row["balcony"] = balcony

        for prefix, value in [
            ("location", location),
            ("area_type", area_type),
            ("availability", availability),
        ]:
            column = f"{prefix}_{value}"
            if column in feature_columns:
                row[column] = 1

        scaled_row = scaler.transform(row)
        price = max(model.predict(scaled_row)[0], 1.0)

        st.subheader("Estimated Price")
        st.success(f"₹ {price:.2f} Lakhs (≈ ₹ {price/100:.2f} Crores)")

else:
    st.info("Add the model.pkl, scaler.pkl and feature_columns.pkl inside the 'output_artifacts' folder.")