import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import gspread
from google.auth import default
from io import StringIO

# Page configuration
st.set_page_config(page_title="🚗 Car Maintenance Predictor", layout="wide")

# Load model and scaler
model = joblib.load("model2.pkl")
scaler = joblib.load("Scaler.pkl")

# Label encoders (use the ones you used during training)
categorical_cols = ['Vehicle_Model', 'Maintenance_History', 'Fuel_Type', 'Transmission_Type', 
                    'Owner_Type', 'Tire_Condition', 'Brake_Condition', 'Battery_Status']
label_encoders = {col: LabelEncoder() for col in categorical_cols}

# Dummy fitting (replace with your saved encoders if available)
for col, classes in {
    'Vehicle_Model': ['truck', 'van', 'bus', 'motorcycle', 'suv', 'car'],
    'Maintenance_History': ['Good', 'Average', 'Poor'],
    'Fuel_Type': ['petrol', 'diesel', 'electric'],
    'Transmission_Type': ['manual', 'automatic'],
    'Owner_Type': ['first', 'second', 'third', 'fourth or above'],
    'Tire_Condition': ['good', 'new', 'worn out'],
    'Brake_Condition': ['good', 'new', 'worn out'],
    'Battery_Status': ['good', 'weak']
}.items():
    label_encoders[col].fit(classes)

# --- UI Elements ---
st.title("🚘 Car Maintenance Prediction App")
st.caption("🔧 *AI-Powered Insight: Predict Potential Maintenance Needs Before They Become Problems*")
st.markdown("---")

# Form Section
with st.form(key='car_input_form'):
    st.subheader("📋 Enter Vehicle Details")

    col1, col2 = st.columns(2)

    with col1:
        vehicle_model = st.selectbox("Vehicle Type", ['truck', 'van', 'bus', 'motorcycle', 'suv', 'car'])
        mileage = st.number_input("Mileage (km/l or km)", min_value=0.0, step=0.1)
        maintenance_history = st.selectbox("Maintenance History", ["Good", "Average", "Poor"])
        reported_issues = st.number_input("Reported Issues", min_value=0, max_value=5)
        vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=10)
        fuel_type = st.selectbox("Fuel Type", ['petrol', 'diesel', 'electric'])
        transmission_type = st.selectbox("Transmission Type", ['manual', 'automatic'])
        engine_size = st.number_input("Engine Size (cc)", min_value=1000.0, max_value=2500.0)

    with col2:
        odometer = st.number_input("Odometer Reading (km)", min_value=0)
        owner_type = st.selectbox("Owner Type", ['first', 'second', 'third', 'fourth or above'])
        insurance_premium = st.number_input("Annual Insurance Premium (₹)", min_value=0.0)
        service_history = st.number_input("Service History Records", min_value=0, max_value=10)
        accident_history = st.number_input("Accident History", min_value=0, max_value=3)
        fuel_efficiency = st.number_input("Fuel Efficiency (km/l)", min_value=0.0)
        tire_condition = st.selectbox("Tire Condition", ['good', 'new', 'worn out'])
        brake_condition = st.selectbox("Brake Condition", ['good', 'new', 'worn out'])
        battery_status = st.selectbox("Battery Status", ['good', 'weak'])

    # Submit button
    submit_button = st.form_submit_button(label="🔮 Predict Maintenance Need")

# --- Processing Logic ---
if submit_button:
    st.markdown("---")
    st.subheader("🧾 Summary of Vehicle Information")

    st.info(f"""
    - **Vehicle Type**: {vehicle_model}
    - **Mileage**: {mileage} km/l
    - **Age**: {vehicle_age} years
    - **Fuel Type**: {fuel_type}
    - **Transmission**: {transmission_type}
    - **Engine Size**: {engine_size} cc
    - **Odometer**: {odometer} km
    - **Owner Type**: {owner_type}
    - **Insurance**: ₹{insurance_premium}
    - **Maintenance History**: {maintenance_history}
    - **Reported Issues**: {reported_issues}
    - **Service Records**: {service_history}
    - **Accidents**: {accident_history}
    - **Efficiency**: {fuel_efficiency} km/l
    - **Tire Condition**: {tire_condition}
    - **Brake Condition**: {brake_condition}
    - **Battery Status**: {battery_status}
    """)

    # Create input dictionary
    input_data = {
        'Vehicle_Model': vehicle_model,
        'Mileage': mileage,
        'Maintenance_History': maintenance_history,
        'Reported_Issues': reported_issues,
        'Vehicle_Age': vehicle_age,
        'Fuel_Type': fuel_type,
        'Transmission_Type': transmission_type,
        'Engine_Size': engine_size,
        'Odometer_Reading': odometer,
        'Owner_Type': owner_type,
        'Insurance_Premium': insurance_premium,
        'Service_History': service_history,
        'Accident_History': accident_history,
        'Fuel_Efficiency': fuel_efficiency,
        'Tire_Condition': tire_condition,
        'Brake_Condition': brake_condition,
        'Battery_Status': battery_status
    }

    # Encode inputs
    encoded_input = []
    for col in categorical_cols:
        encoded_input.append(label_encoders[col].transform([input_data[col]])[0])
    for col in input_data:
        if col not in categorical_cols:
            encoded_input.append(input_data[col])

    input_array = np.array(encoded_input).reshape(1, -1)
    # input_array = scaler.transform(input_array)  # Optional: scale if needed

    # Predict and calculate probabilities
    prediction = model.predict(input_array)[0]
    prediction_probabilities = model.predict_proba(input_array)[0]

    # Convert to Yes/No
    maintenance_status = "Yes" if prediction == 1 else "No"

    # --- Display Prediction Result ---
    st.markdown("### ✅ Prediction Result")
    st.success(f"🔧 **Predicted Maintenance Needed?** → **{maintenance_status}**")
    st.markdown("ℹ️ *Ensure timely checks to keep your vehicle in peak condition.*")

    # --- Show Prediction Probability Chart ---
    st.markdown("### 📊 Prediction Probability")
    labels = ["No Maintenance", "Maintenance Needed"]
    fig, ax = plt.subplots()
    ax.bar(labels, prediction_probabilities, color=["green", "red"])
    ax.set_ylim([0, 1])
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probabilities")
    st.pyplot(fig)

    # --- Maintenance Tips ---
    st.markdown("### 🛠️ Maintenance Tips")
    if prediction == 1:
        st.write("🔴 **Maintenance is needed!** It’s essential to address any underlying issues to prevent further damage.")
    else:
        st.write("🟢 **No maintenance required**. Your vehicle is in good condition, but regular checks are still recommended.")

    # --- Save Prediction Option ---
    st.markdown("### 💾 Save Prediction")
    save_option = st.selectbox("Choose where to save the prediction:", ["None", "Google Sheets", "Local CSV"])

    if save_option == "Google Sheets":
        st.write("To save to Google Sheets, you need to authenticate your Google account first.")
        # Add Google Sheets save functionality here using gspread API.
    elif save_option == "Local CSV":
        st.download_button("Download Prediction as CSV", data=pd.DataFrame([input_data]).to_csv(), file_name="car_maintenance_prediction.csv", mime="text/csv")
