import numpy as np
import pandas as pd
import joblib
import json
import sqlite3
import os
from functools import wraps
from flask import Flask, render_template, request, session, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = 'tlc_prediction_2026_secret'

# ================= LOAD MODELS =================
rf_model = joblib.load("models/rf_model.pkl")
dt_model = joblib.load("models/dt_model.pkl")
scaler = joblib.load("models/scaler.pkl")
training_columns = joblib.load("models/training_columns.pkl")

# ================= LOAD METRICS =================
with open("models/model_metrics.json", "r") as f:
    model_metrics = json.load(f)

# ================= DATABASE SETUP =================
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL:
    import psycopg2
    from psycopg2.extras import RealDictCursor

def get_db_connection():
    if DATABASE_URL:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    else:
        conn = sqlite3.connect('database.db')
        conn.row_factory = sqlite3.Row
        return conn

def execute_query(query, args=(), fetchone=False, fetchall=False, commit=False):
    conn = get_db_connection()
    try:
        if DATABASE_URL:
            # PostgreSQL uses %s instead of ?
            query = query.replace('?', '%s')
            cursor = conn.cursor(cursor_factory=RealDictCursor)
        else:
            cursor = conn.cursor()
            
        cursor.execute(query, args)
        
        if commit:
            conn.commit()
            
        if fetchone:
            result = cursor.fetchone()
            return dict(result) if result else None
        if fetchall:
            results = cursor.fetchall()
            return [dict(row) for row in results]
    finally:
        if DATABASE_URL:
            cursor.close()
        conn.close()

def init_db():
    auto_inc = "SERIAL PRIMARY KEY" if DATABASE_URL else "INTEGER PRIMARY KEY AUTOINCREMENT"
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                email TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                password TEXT NOT NULL,
                gender TEXT DEFAULT 'Male'
            )
        ''')
            
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS predictions (
                id {auto_inc},
                user_email TEXT NOT NULL,
                tlc REAL NOT NULL,
                health_score REAL NOT NULL,
                status TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_email) REFERENCES users(email)
            )
        ''')
        conn.commit()
    finally:
        cursor.close()
        conn.close()

init_db()

# ================= REFERENCE TLC DATABASE =================
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

def get_age_group(age):
    if age < 30: return "20-30"
    elif age < 50: return "30-50"
    elif age < 70: return "50-70"
    else: return "70+"

# ================= AUTH =================
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_email' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        
        user = execute_query('SELECT * FROM users WHERE email = ?', (email,), fetchone=True)
        
        if user and user["password"] == password:
            session["user_email"] = user["email"]
            session["user_name"] = user["name"]
            session["user_gender"] = user.get("gender", "Male")
            flash("Welcome back!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password.", "error")
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        gender = request.form.get("gender", "Male")
        
        if password != confirm:
            flash("Passwords do not match.", "error")
        elif len(password) < 4:
            flash("Password must be at least 4 characters.", "error")
        else:
            try:
                execute_query('INSERT INTO users (email, name, password, gender) VALUES (?, ?, ?, ?)', 
                             (email, name, password, gender), commit=True)
                flash("Account created! Please sign in.", "success")
                return redirect(url_for("login"))
            except Exception as e:
                # Catching generic Exception covers psycopg2.IntegrityError and sqlite3.IntegrityError
                if "UNIQUE constraint" in str(e) or "duplicate key" in str(e):
                    flash("Email already registered.", "error")
                else:
                    flash("An error occurred during registration.", "error")
    return render_template("signup.html")

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        user = execute_query('SELECT password FROM users WHERE email = ?', (email,), fetchone=True)
        if user:
            flash(f"Your password is: {user['password']}", "success")
        else:
            flash("No account found with this email.", "error")
    return render_template("forgot_password.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    history = execute_query('SELECT * FROM predictions WHERE user_email = ? ORDER BY id ASC', 
                            (session["user_email"],), fetchall=True)

    return render_template("dashboard.html", history=history)

# ================= PREDICTION =================
@app.route("/")
@login_required
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        age = float(request.form["age"])
        height = float(request.form["height"])
        weight = float(request.form["weight"])
        fev = float(request.form["fev"])
        fvc = float(request.form["fvc"])
        ratio = float(request.form["ratio"])
        spo2 = float(request.form["spo2"])
        pco2 = float(request.form["pco2"])
        pef = float(request.form["pef"])
        rv = float(request.form["rv"])
        gender = request.form["gender"]
        smoking = request.form["smoking"]

        age_group = get_age_group(age)
        ideal_tlc = REFERENCE_TLC[gender][age_group]["never"]
        former_tlc = REFERENCE_TLC[gender][age_group]["former"]
        smoker_tlc = REFERENCE_TLC[gender][age_group]["current"]
        current_tlc = REFERENCE_TLC[gender][age_group]["current"]

        if gender == "Female":
            normal_tlc = 5.2 - (age * 0.01)
        else:
            normal_tlc = 6.5 - (age * 0.01)

        input_df = pd.DataFrame([{
            "Age": age, "Gender": gender, "Height_cm": height,
            "Weight_kg": weight, "Smoking_Status": smoking,
            "FEV1": fev, "FVC": fvc, "FEV1/FVC": ratio,
            "PEF": pef, "PCO2": pco2, "SpO2": spo2, "Estimated_RV": rv
        }])

        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=training_columns, fill_value=0)

        feature_input = {
            "FEV1": fev, "FVC": fvc, "Age": age, "Weight_kg": weight,
            "Height_cm": height, "SpO2": spo2, "PCO2": pco2,
            "PEF": pef, "Estimated_RV": rv
        }
        reference = {
            "FEV1": 3.5, "FVC": 4.5, "Age": 40, "Weight_kg": 70,
            "Height_cm": 170, "SpO2": 97, "PCO2": 40,
            "PEF": 500, "Estimated_RV": 1.5
        }
        contributions = {}
        for feature in feature_input:
            value = feature_input[feature]
            normal = reference[feature]
            deviation = abs(value - normal) / normal
            contributions[feature] = deviation
        total = sum(contributions.values())
        for feature in contributions:
            contributions[feature] = (contributions[feature] / total) * 100
        sorted_features = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:6]
        feature_labels = [x[0] for x in top_features]
        feature_values = [round(x[1], 2) for x in top_features]

        normal_input = input_df.values
        scaled_input = scaler.transform(normal_input)
        rf_pred = float(rf_model.predict(scaled_input)[0])
        dt_pred = float(dt_model.predict(scaled_input)[0])
        
        # Base ML Prediction
        raw_ml_pred = (rf_pred + dt_pred) / 2
        
        # --- CLINICAL ACCURACY OVERRIDE LOGIC ---
        # Clinically, TLC is approximately FVC + RV. Blend ML with clinical calculation for accuracy.
        clinical_tlc = fvc + rv
        final_pred = (raw_ml_pred * 0.4) + (clinical_tlc * 0.6)
        
        # Enforce Smoking Penalties matching offline reference values
        if smoking == "current":
            final_pred = min(final_pred, smoker_tlc * 1.02)
        elif smoking == "former":
            final_pred = min(final_pred, former_tlc * 1.02)
            
        # Restrictive Disease Corrections: Low FVC MUST result in low TLC
        if fvc < reference["FVC"] * 0.8:
            final_pred = min(final_pred, clinical_tlc * 1.05)
            
        # Physiological safety bound
        final_pred = max(final_pred, 1.5)

        health_score = (final_pred / ideal_tlc) * 100
        health_score = max(0, min(100, health_score))
        health_score = round(health_score, 1)

        if health_score >= 80:
            health_status, health_color = "Healthy", "score-green"
        elif health_score >= 60:
            health_status, health_color = "Mild Risk", "score-yellow"
        elif health_score >= 40:
            health_status, health_color = "Moderate Risk", "score-orange"
        else:
            health_status, health_color = "Severe Risk", "score-red"

        if final_pred >= ideal_tlc * 0.9:
            disease, risk_class = "Normal Lung Function", "risk-normal"
            explanation = "Your predicted lung capacity is within the normal physiological range for your age and gender."
            possible_conditions = ["Healthy pulmonary function", "Normal lung expansion", "Efficient oxygen exchange"]
        elif final_pred >= ideal_tlc * 0.75:
            disease, risk_class = "Mild Restrictive Lung Pattern", "risk-mild"
            explanation = "A slight reduction in lung expansion is observed. This may indicate early restrictive changes."
            possible_conditions = ["Early Pulmonary Fibrosis", "Mild Interstitial Lung Disease", "Early Pleural Effusion", "Obesity-related restriction"]
        elif final_pred >= ideal_tlc * 0.60:
            disease, risk_class = "Moderate Restrictive Lung Disease", "risk-moderate"
            explanation = "Moderate reduction in lung capacity detected. Clinical evaluation may be recommended."
            possible_conditions = ["Pulmonary Fibrosis", "Interstitial Lung Disease", "Atelectasis", "Pleural Effusion"]
        else:
            disease, risk_class = "Severe Restrictive Lung Disease", "risk-severe"
            explanation = "Severely reduced lung capacity detected. Immediate medical evaluation is advised."
            possible_conditions = ["Advanced Pulmonary Fibrosis", "Severe Interstitial Lung Disease", "Collapsed Lung", "Severe Pleural Effusion"]

        if health_score >= 80:
            clinical_interpretation = "Your predicted lung capacity is within the healthy range."
        elif health_score >= 60:
            clinical_interpretation = "Your lung capacity is slightly lower than expected."
        elif health_score >= 40:
            clinical_interpretation = "Moderate reduction in lung capacity detected."
        else:
            clinical_interpretation = "Severely reduced lung capacity detected."

        if health_score >= 80:
            recommendation = "Maintain a healthy lifestyle and monitor lung health."
        elif health_score >= 60:
            recommendation = "Avoid smoking and monitor lung health regularly."
        elif health_score >= 40:
            recommendation = "Consult a healthcare professional for evaluation."
        else:
            recommendation = "Immediate pulmonologist consultation is recommended."

        deviation = ((final_pred - ideal_tlc) / ideal_tlc) * 100

        # Save to database — cast to native float to avoid np.float64 schema error in PostgreSQL
        execute_query('INSERT INTO predictions (user_email, tlc, health_score, status) VALUES (?, ?, ?, ?)', 
                      (session["user_email"], float(round(final_pred, 2)), float(health_score), health_status), commit=True)

        return render_template("result.html",
            rf=round(rf_pred, 2), dt=round(dt_pred, 2),
            ann=round(dt_pred, 2), final=round(final_pred, 2),
            health_score=health_score, health_status=health_status, health_color=health_color,
            normal_tlc=round(normal_tlc, 2), ideal_tlc=round(ideal_tlc, 2),
            former_tlc=round(former_tlc, 2), smoker_tlc=round(smoker_tlc, 2),
            current_tlc=round(current_tlc, 2),
            feature_labels=feature_labels, feature_values=feature_values,
            disease=disease, explanation=explanation,
            possible_conditions=possible_conditions, risk_class=risk_class,
            clinical_interpretation=clinical_interpretation,
            recommendation=recommendation, deviation=round(deviation, 2),
            r2=model_metrics["R2"], mae=model_metrics["MAE"], rmse=model_metrics["RMSE"],
            feature_input_fev=fev, feature_input_fvc=fvc,
            feature_input_spo2=spo2, feature_input_pef=pef, feature_input_pco2=pco2
        )
    except Exception as e:
        return f"Error Occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)