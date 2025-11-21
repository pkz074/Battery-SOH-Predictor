from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Gemini Setup
genai.configure(api_key="YOUR_GEMINI_API_KEY")
gemini_model = genai.GenerativeModel("gemini-2.5-flash-lite")

def ask_gemini(question):
    try:
        response = gemini_model.generate_content(question)
        return response.text
    except Exception as e:
        return f"[ERROR] Gemini API failed: {e}"

def detect_soh_intent(user_input):
    detection_prompt = (
        f'A user enters: "{user_input}"\n'
        "Is the user requesting a battery State of Health (SOH) prediction using cell voltage values? "
        "Reply with YES or NO only."
    )
    result = ask_gemini(detection_prompt).strip().lower()
    return result.startswith("yes")

# Load and train model
file_path = "PulseBat Dataset.xlsx"
sheet_name = "SOC ALL"
try:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = (df.columns.str.strip()
                  .str.replace(r'\(.*\)', '', regex=True)
                  .str.replace(r'[_\.\s]+', '', regex=True).str.upper())
    required_cols = ["NO", "SOC", "SOH"] + [f"U{i}" for i in range(1, 22)]
    feature_cols = [f"U{i}" for i in range(1, 22)]
    df_original = df.reset_index(drop=True)
    X = df_original[feature_cols]
    y = df_original["SOH"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    intercept = model.intercept_
    coefficients = model.coef_
    equation = f"SOH = {intercept:.6f}"
    for i, coef in enumerate(coefficients, start=1):
        sign = " + " if coef >= 0 else " - "
        equation += f"{sign}{abs(coef):.6f}*U{i}"
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def get_battery_health_prediction(input_values, threshold=0.6):
    input_df = pd.DataFrame([input_values], columns=[f"U{i}" for i in range(1, 22)])
    predicted_soh = model.predict(input_df)[0]
    status = "Healthy" if predicted_soh >= threshold else "Unhealthy"
    return predicted_soh, status

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({'response': 'Please enter a message.', 'type': 'error'})
    
    # Check if prediction intent
    if detect_soh_intent(user_message):
        return jsonify({
            'response': 'Please enter 21 voltage values (U1-U21) separated by spaces:',
            'type': 'prediction_request'
        })
    
    # General query - ask Gemini
    response = ask_gemini(user_message)
    return jsonify({'response': response, 'type': 'general'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    values_str = data.get('values', '').strip()
    threshold = float(data.get('threshold', 0.6))
    
    try:
        values = [float(v) for v in values_str.split()]
    except ValueError:
        return jsonify({'response': 'ERROR: Please enter only numeric values separated by spaces.', 'type': 'error'})
    
    if len(values) != 21:
        return jsonify({'response': f'ERROR: You entered {len(values)} values. Please enter exactly 21 values.', 'type': 'error'})
    
    soh, health = get_battery_health_prediction(values, threshold=threshold)
    
    result = f"""
**Prediction Result**

**Linear Regression Equation:**
{equation}

**Predicted SOH:** {soh:.4f}
**Status:** {health}
**Threshold:** {threshold}
    """
    
    return jsonify({'response': result, 'type': 'prediction_result'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
