import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ======================================
# 1. READ CSV — auto detect separator
# ======================================
file_path = "winequality-red.csv"
data = pd.read_csv(file_path, sep=None, engine="python")

print("\n📌 Dataset loaded successfully")
print(data.head())
print("\n📌 Detected columns:")
print(list(data.columns))

# ======================================
# 2. Normalize column names (lowercase & strip spaces)
# ======================================
data.columns = [c.strip().lower() for c in data.columns]

# Check if quality exists
if "quality" not in data.columns:
    raise Exception("❌ No 'quality' column found. Check dataset file.")

# ======================================
# 3. Split features and target
# ======================================
X = data.drop("quality", axis=1)
y = data["quality"]

# ======================================
# 4. Train–Test Split
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ======================================
# 5. Scale Input Features
# ======================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ======================================
# 6. Model
# ======================================
model = RandomForestRegressor(
    n_estimators=250,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# ======================================
# 7. Evaluate
# ======================================
pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\n📊 Model Performance:")
print("RMSE :", rmse)
print("R²   :", r2)

# ======================================
# 8. Save model + scaler
# ======================================
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Model saved to model.pkl")
print("✅ Scaler saved to scaler.pkl")
