
import streamlit as st
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Car Maintenance Predictor", layout="wide")

# Define car brands and models (same as your existing dictionary)

car_brands = {
    "Toyota": [
        "Toyota Fortuner", "Toyota Innova", "Toyota Corolla Altis", "Toyota Camry",
        "Toyota Land Cruiser", "Toyota Hilux", "Toyota Etios", "Toyota Prius"
    ],
    "Honda": [
        "Honda Civic", "Honda CR-V", "Honda Accord", "Honda City", "Honda Jazz",
        "Honda WR-V", "Honda Amaze", "Honda Brio"
    ],
    "Hyundai": [
        "Hyundai Creta", "Hyundai Tucson", "Hyundai Elantra", "Hyundai i20",
        "Hyundai Verna", "Hyundai Venue", "Hyundai Kona", "Hyundai Santro"
    ],
    "Mahindra": [
        "Mahindra XUV500", "Mahindra Thar", "Mahindra Scorpio", "Mahindra Bolero",
        "Mahindra Marazzo", "Mahindra Alturas G4", "Mahindra XUV700", "Mahindra TUV300"
    ],
    "Ford": [
        "Ford EcoSport", "Ford Endeavour", "Ford Figo", "Ford Aspire", "Ford Mustang",
        "Ford Focus", "Ford Ranger"
    ],
    "BMW": [
        "BMW X5", "BMW X3", "BMW Z4", "BMW 3 Series", "BMW 5 Series",
        "BMW X1", "BMW 7 Series", "BMW 8 Series"
    ],
    "Mercedes-Benz": [
        "Mercedes-Benz GLC", "Mercedes-Benz G-Class", "Mercedes-Benz A-Class", "Mercedes-Benz C-Class",
        "Mercedes-Benz E-Class", "Mercedes-Benz S-Class", "Mercedes-Benz CLA", "Mercedes-Benz SLS AMG"
    ],
    "Audi": [
        "Audi Q7", "Audi A6", "Audi Q5", "Audi A4", "Audi A3", "Audi Q3", "Audi RS5", "Audi A8"
    ],
    "Nissan": [
        "Nissan X-Trail", "Nissan Altima", "Nissan Pathfinder", "Nissan Kicks", "Nissan Sunny",
        "Nissan Leaf", "Nissan Micra", "Nissan Rogue"
    ],
    "Volkswagen": [
        "Volkswagen Polo", "Volkswagen Jetta", "Volkswagen Passat", "Volkswagen Tiguan", "Volkswagen Beetle",
        "Volkswagen Golf", "Volkswagen Arteon"
    ],
    "Jaguar": [
        "Jaguar F-Type", "Jaguar XE", "Jaguar XF", "Jaguar F-PACE", "Jaguar I-PACE"
    ],
    "Porsche": [
        "Porsche 911", "Porsche Cayenne", "Porsche Macan", "Porsche Taycan", "Porsche Panamera"
    ],
    "Chevrolet": [
        "Chevrolet Cruze", "Chevrolet Spark", "Chevrolet Malibu", "Chevrolet Trailblazer",
        "Chevrolet Beat", "Chevrolet Captiva"
    ],
    "Kia": [
        "Kia Seltos", "Kia Sonet", "Kia Carnival", "Kia K5", "Kia Stinger", "Kia Sportage"
    ],
    "Renault": [
        "Renault Kwid", "Renault Duster", "Renault Triber", "Renault Lodgy", "Renault Captur"
    ],
    "Skoda": [
        "Skoda Superb", "Skoda Octavia", "Skoda Kushaq", "Skoda Rapid", "Skoda Fabia"
    ],
    "Tata": [
        "Tata Nexon", "Tata Harrier", "Tata Tiago", "Tata Tigor", "Tata Altroz", "Tata Safari"
    ],
    "Fiat": [
        "Fiat Linea", "Fiat Punto", "Fiat Panda", "Fiat 500X"
    ],
    "Land Rover": [
        "Land Rover Range Rover", "Land Rover Discovery", "Land Rover Defender"
    ],
    "Mitsubishi": [
        "Mitsubishi Outlander", "Mitsubishi Pajero", "Mitsubishi Lancer"
    ],
    "Volvo": [
        "Volvo XC60", "Volvo XC90", "Volvo S60", "Volvo S90", "Volvo V90", "Volvo XC40"
    ],
    "Suzuki": [
        "Suzuki Swift", "Suzuki Dzire", "Suzuki Vitara Brezza", "Suzuki Baleno", "Suzuki Alto", "Suzuki Celerio"
    ],
    "Peugeot": [
        "Peugeot 208", "Peugeot 308", "Peugeot 5008", "Peugeot 2008"
    ],
    "Lexus": [
        "Lexus RX", "Lexus NX", "Lexus LX", "Lexus ES", "Lexus UX", "Lexus GS"
    ],
    "Acura": [
        "Acura MDX", "Acura RDX", "Acura ILX", "Acura TLX", "Acura NSX"
    ]
}

# Load the pre-trained model and scaler
model = joblib.load("model2.pkl")
scaler = joblib.load("Scaler.pkl")

st.title("Car Maintenance Prediction App")
st.caption("Precision-Driven Insights for Predictive Car Maintenance – Minimize Downtime, Maximize Performance")
st.divider()

# Form for user input
with st.form(key='car_input_form'):
    st.subheader("Enter Car Details")
    
    # Vehicle model input (dynamically updated based on brand)
    brand = st.selectbox("Car Brand", list(car_brands.keys()))
    vehicle_model = st.selectbox("Vehicle Model", car_brands[brand])
    
    # Other input fields
    maintenance_history = st.selectbox("Maintenance History", ["Regular", "Irregular"])
    fuel_type = st.selectbox("Fuel Type", ["Diesel", "Petrol", "Electric"])
    transmission_type = st.selectbox("Transmission Type", ["Manual", "Automatic"])
    owner_type = st.selectbox("Owner Type", ["First", "Second", "Third"])
    tire_condition = st.selectbox("Tire Condition", ["Good", "Average", "Poor"])
    brake_condition = st.selectbox("Brake Condition", ["Good", "Average", "Poor"])
    battery_status = st.selectbox("Battery Status", ["Good", "Replace Soon", "Replace Immediately"])

    submit_button = st.form_submit_button(label="Predict Car Maintenance")

if submit_button:
    # Display submitted car details
    st.write("Car Details Submitted:")
    st.write(f"Car Brand: {brand}")
    st.write(f"Vehicle Model: {vehicle_model}")
    st.write(f"Maintenance History: {maintenance_history}")
    st.write(f"Fuel Type: {fuel_type}")
    st.write(f"Transmission Type: {transmission_type}")
    st.write(f"Owner Type: {owner_type}")
    st.write(f"Tire Condition: {tire_condition}")
    st.write(f"Brake Condition: {brake_condition}")
    st.write(f"Battery Status: {battery_status}")

    # Initialize label encoder
    label_encoder = LabelEncoder()

    # Encode categorical inputs
    fuel_type_encoded = label_encoder.fit_transform([fuel_type])[0]
    transmission_type_encoded = label_encoder.fit_transform([transmission_type])[0]
    owner_type_encoded = label_encoder.fit_transform([owner_type])[0]
    maintenance_history_encoded = label_encoder.fit_transform([maintenance_history])[0]
    tire_condition_encoded = label_encoder.fit_transform([tire_condition])[0]
    brake_condition_encoded = label_encoder.fit_transform([brake_condition])[0]
    battery_status_encoded = label_encoder.fit_transform([battery_status])[0]

    # Prepare feature vector
    input_features = [
        fuel_type_encoded, 
        transmission_type_encoded, 
        owner_type_encoded, 
        maintenance_history_encoded,
        tire_condition_encoded, 
        brake_condition_encoded, 
        battery_status_encoded
    ]
    
    # Scale the features if the model was trained with scaling
    input_features_scaled = scaler.transform([input_features])

    # Make prediction with the trained Decision Tree model
    prediction = model.predict(input_features_scaled)

    # Display the prediction result
    st.subheader("Prediction Result")
    st.write(f"Predicted Maintenance Need: {prediction[0]}")
