#!/usr/bin/env python3

# ============================================================
# Imports
# ============================================================
import os
import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ============================================================
# Gemini Setup
# ============================================================
genai.configure(api_key="AIzaSyBxTzSoGzWldeVRPURLb2Xvb2Pi2WEpt1c")

gemini_model = genai.GenerativeModel("gemini-3-pro-preview")

def ask_gemini(question):
    """Send a user question to Gemini and return natural-language answer."""
    try:
        response = gemini_model.generate_content(question)
        return response.text
    except Exception as e:
        return f"[ERROR] Gemini API failed: {e}"


# ============================================================
# Load Dataset
# ============================================================
file_path = "PulseBat Dataset.xlsx"
sheet_name = "SOC ALL"

print("[INFO] Loading dataset...")
df = pd.read_excel(file_path, sheet_name=sheet_name)
print(f"[INFO] Loaded '{sheet_name}' with shape {df.shape}")

# Normalize column names
df.columns = (
    df.columns
    .str.strip()
    .str.replace(r'\(.*\)', '', regex=True)
    .str.replace(r'[_\.\s]+', '', regex=True)
    .str.upper()
)

print("[INFO] Normalized columns:")
print(df.columns.tolist())

# ============================================================
# Validate required columns
# ============================================================
required_cols = ["NO", "SOC", "SOH"] + [f"U{i}" for i in range(1, 22)]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# ============================================================
# Feature selection
# ============================================================
feature_cols = [f"U{i}" for i in range(1, 22)]

# ============================================================
# Sorting options
# ============================================================
def sort_rows(df, method):
    if method == "original":
        return df.reset_index(drop=True)
    elif method == "by_ID":
        return df.sort_values("NO").reset_index(drop=True)
    elif method == "by_SOC":
        return df.sort_values("SOC").reset_index(drop=True)
    elif method == "random":
        return df.sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        raise ValueError(f"Unknown sorting method: {method}")


sorting_methods = ["original", "by_ID", "by_SOC", "random"]

# ============================================================
# Train & Evaluate Models
# ============================================================
results_summary = []

for method in sorting_methods:
    print(f"\n==============================")
    print(f" Training with '{method}' sorting ")
    print(f"==============================")

    dset = sort_rows(df, method)
    X = dset[feature_cols]
    y = dset["SOH"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")

    results_summary.append([method, r2, mse, mae])

results_df = pd.DataFrame(results_summary, columns=["Sorting Method", "R² Score", "MSE", "MAE"])
print("\n------ Model Performance Comparison ------")
print(results_df.to_string(index=False))

# ============================================================
# Final model (original sorting)
# ============================================================
df_original = sort_rows(df, "original")
X = df_original[feature_cols]
y = df_original["SOH"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Print regression equation
intercept = model.intercept_
coefficients = model.coef_
equation = f"SOH = {intercept:.6f}"
for i, coef in enumerate(coefficients, start=1):
    sign = " + " if coef >= 0 else " - "
    equation += f"{sign}{abs(coef):.6f}*U{i}"

print("\n------ Linear Regression Equation ------")
print(equation)


# ============================================================
# Battery SOH Prediction Function
# ============================================================
def get_battery_health_prediction(input_values, threshold=0.6):
    if len(input_values) != 21:
        return "ERROR: You must enter exactly 21 voltage values."

    input_df = pd.DataFrame([input_values], columns=[f"U{i}" for i in range(1, 22)])
    predicted_soh = model.predict(input_df)[0]

    status = "Healthy" if predicted_soh >= threshold else "Unhealthy"

    return (
        f"Predicted SOH: {predicted_soh:.4f}\n"
        f"Status: {status} (threshold = {threshold})"
    )


# ============================================================
# Interactive Chatbot
# ============================================================
print("\n\n=================================================")
print("   Battery Health Chatbot with Gemini AI")
print("=================================================")
print("Type:")
print("  • 'predict' → Enter U1–U21 values for SOH prediction")
print("  • Ask any question → Gemini will answer")
print("  • 'exit' → Quit")
print("=================================================\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    elif user_input.lower() == "predict":
        print("\nEnter 21 voltage values (U1–U21):")
        values = []
        for i in range(1, 22):
            val = float(input(f"U{i}: "))
            values.append(val)

        print("\n--- Prediction Result ---")
        print(get_battery_health_prediction(values))
        print("-------------------------\n")

    else:
        print("\n--- Gemini Answer ---")
        print(ask_gemini(user_input))
        print("---------------------\n")
