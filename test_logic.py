import pandas as pd
import joblib
import numpy as np

# Load models and preprocessing
rf_model = joblib.load("models/rf_model.pkl")
dt_model = joblib.load("models/dt_model.pkl")
scaler = joblib.load("models/scaler.pkl")
training_columns = joblib.load("models/training_columns.pkl")

def get_age_group(age):
    if age < 30: return "20-30"
    elif age < 50: return "30-50"
    elif age < 70: return "50-70"
    else: return "70+"

REFERENCE_TLC = {
    "Male": {
        "20-30": {"never": 6.00, "former": 5.33, "current": 4.88},
        "30-50": {"never": 5.86, "former": 5.11, "current": 4.86},
        "50-70": {"never": 5.64, "former": 5.01, "current": 4.87},
        "70+":   {"never": 5.51, "former": 4.97, "current": 4.52}
    },
    "Female": {
        "20-30": {"never": 5.52, "former": 4.81, "current": 4.55},
        "30-50": {"never": 5.27, "former": 4.81, "current": 4.49},
        "50-70": {"never": 5.09, "former": 4.54, "current": 4.54},
        "70+":   {"never": 5.03, "former": 4.55, "current": 4.36}
    }
}

def test_prediction(name, age, gender, height, weight, smoking, fev, fvc, ratio, pef, pco2, spo2, rv):
    input_df = pd.DataFrame([{
        "Age": age, "Gender": gender, "Height_cm": height,
        "Weight_kg": weight, "Smoking_Status": smoking,
        "FEV1": fev, "FVC": fvc, "FEV1/FVC": ratio,
        "PEF": pef, "PCO2": pco2, "SpO2": spo2, "Estimated_RV": rv
    }])

    # Get reference
    age_group = get_age_group(age)
    ideal_tlc = REFERENCE_TLC[gender][age_group]["never"]
    expected_tlc = REFERENCE_TLC[gender][age_group][smoking] if smoking in ["current", "former"] else ideal_tlc

    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=training_columns, fill_value=0)

    normal_input = input_df.values
    scaled_input = scaler.transform(normal_input)
    
    rf_pred = rf_model.predict(normal_input)[0]
    dt_pred = dt_model.predict(normal_input)[0]
    raw_ml_pred = (rf_pred + dt_pred) / 2
    
    # Apply logic fix idea
    clinical_tlc = fvc + rv
    blended_pred = (raw_ml_pred * 0.4) + (clinical_tlc * 0.6)
    
    # Penalize smoking if ML missed it
    if smoking == "current":
        blended_pred = min(blended_pred, expected_tlc * 1.05)
        
    print(f"--- Case: {name} ---")
    print(f"Inputs: FVC={fvc}, FEV1={fev}, RV={rv}, Smoke={smoking}, Ideal={ideal_tlc}, Expected(Smoke)={expected_tlc}")
    print(f"Raw ML Pred: {raw_ml_pred:.2f}")
    print(f"Clinical TLC (FVC+RV): {clinical_tlc:.2f}")
    print(f"Adjusted Hybrid Pred: {blended_pred:.2f}\n")


test_prediction("Healthy Male", 25, "Male", 180, 75, "never", 4.0, 5.0, 0.8, 500, 40, 98, 1.5)
test_prediction("Heavy Smoker Male", 45, "Male", 175, 80, "current", 2.0, 3.0, 0.66, 300, 45, 92, 2.5)
test_prediction("Severe Restrictive Female", 65, "Female", 160, 60, "never", 1.5, 2.0, 0.75, 200, 38, 90, 1.0)
test_prediction("Healthy Female", 30, "Female", 165, 62, "never", 3.0, 4.0, 0.75, 400, 40, 98, 1.5)
