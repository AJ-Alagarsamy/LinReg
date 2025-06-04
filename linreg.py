import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# --- Data Loading & Validation ---
try:
    df = pd.read_csv("one.csv")
    print("Successfully loaded one.csv")
    print("Available columns:", df.columns.tolist())
except FileNotFoundError:
    print("Error: one.csv not found in the current directory")
    exit()

# Check required columns
required_columns = ["IV", "Spot", "HV", "Strike"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Error: Missing required columns - {missing_columns}")
    exit()

# --- Data Cleaning ---
def clean_iv_column(iv_series):
    """Convert IV column to decimal format (e.g., '50%' -> 0.50)."""
    if iv_series.dtype == object:
        if iv_series.str.contains('%').any():
            return iv_series.str.replace('%', '').astype(float) / 100
        else:
            return pd.to_numeric(iv_series, errors='coerce')
    elif iv_series.max() > 1:  # Assume IV is in percentage (e.g., 50 for 50%)
        return iv_series / 100
    return iv_series

df["IV"] = clean_iv_column(df["IV"])
if df["IV"].isna().any():
    print("Warning: Some IV values could not be converted to numbers. Dropping invalid rows.")
    df = df.dropna(subset=["IV"])

# --- Feature Selection ---
available_features = ["Spot", "HV", "Strike"]
optional_features = ["Daily_Change", "Weekly_Change"]

X_columns = available_features.copy()
for feature in optional_features:
    if feature in df.columns:
        X_columns.append(feature)
    else:
        print(f"Note: {feature} not found - proceeding without it")

X = df[X_columns]
y = df["IV"]

# --- Model Training & Evaluation ---
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

cv = KFold(n_splits=5, shuffle=True, random_state=42)
r2_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2')

print("\nModel Performance:")
print(f"Cross-validated R² scores: {r2_scores}")
print(f"Average R²: {r2_scores.mean():.4f}")

# --- Plot Actual vs Predicted IV ---
pipeline.fit(X, y)
y_pred = pipeline.predict(X)

plt.figure(figsize=(12, 6))
plt.plot(y.values, label="Actual IV", marker='o', alpha=0.7)
plt.plot(y_pred, label="Predicted IV", marker='x', alpha=0.7)
plt.legend()
plt.title("Actual vs Predicted Implied Volatility")
plt.xlabel("Data Point Index")
plt.ylabel("IV")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Feature Importance ---
coefficients = pd.DataFrame({
    'Feature': X_columns,
    'Coefficient': pipeline.named_steps['regressor'].coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nFeature Importance:")
print(coefficients)

# --- Prediction Function ---
def predict_iv_for_strikes(model, current_spot, current_hv, strikes):
    """Predict IV for new strikes at current market conditions."""
    # Validate inputs
    if not isinstance(strikes, (list, np.ndarray)):
        strikes = [strikes]
    
    # Create input DataFrame with available features
    input_data = pd.DataFrame({
        'Spot': [current_spot] * len(strikes),
        'HV': [current_hv] * len(strikes),
        'Strike': strikes
    })
    
    # Add optional features if they were used in training
    for feature in optional_features:
        if feature in X_columns:
            input_data[feature] = df[feature].mean()  # Use historical average
    
    return model.predict(input_data)

# --- Interactive Prediction ---
try:
    current_spot = float(input("Enter current underlying price: "))
    current_hv = float(input("Enter current historical volatility (e.g., 0.20 for 20%): "))
    new_strikes = list(map(float, input("Enter strikes to predict (comma-separated): ").split(',')))
    
    predicted_ivs = predict_iv_for_strikes(pipeline, current_spot, current_hv, new_strikes)
    
    print("\nPredicted Implied Volatilities:")
    for strike, iv in zip(new_strikes, predicted_ivs):
        print(f"Strike {strike}: {iv:.2%}")
    
    # --- Plot Volatility Smile ---
    plt.figure(figsize=(10, 6))
    plt.plot(new_strikes, predicted_ivs, 'bo-', label='Predicted IV')
    plt.axvline(current_spot, color='r', linestyle='--', label='Current Spot')
    plt.title("Predicted Volatility Smile")
    plt.xlabel("Strike Price")
    plt.ylabel("Implied Volatility")
    plt.legend()
    plt.grid(True)
    plt.show()

except ValueError:
    print("Error: Invalid input. Please enter numbers only.")