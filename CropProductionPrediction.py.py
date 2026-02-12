# CropProductionPrediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Step 1: Create Sample Dataset
# -----------------------------

data = {
    "Crop": ["Rice", "Wheat", "Maize", "Rice", "Wheat", "Maize", "Rice", "Wheat"],
    "Season": ["Kharif", "Rabi", "Kharif", "Rabi", "Kharif", "Rabi", "Kharif", "Rabi"],
    "Cost_of_Cultivation": [20000, 18000, 15000, 21000, 17000, 16000, 22000, 17500],
    "Production": [50, 45, 40, 55, 43, 42, 60, 44]
}

df = pd.DataFrame(data)

# -----------------------------
# Step 2: Encode Categorical Data
# -----------------------------

le_crop = LabelEncoder()
le_season = LabelEncoder()

df["Crop"] = le_crop.fit_transform(df["Crop"])
df["Season"] = le_season.fit_transform(df["Season"])

# -----------------------------
# Step 3: Split Data
# -----------------------------

X = df[["Crop", "Season", "Cost_of_Cultivation"]]
y = df["Production"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 4: Train Model
# -----------------------------

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Step 5: Evaluate Model
# -----------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Model Evaluation:")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# -----------------------------
# Step 6: Make Sample Prediction
# -----------------------------

sample_input = pd.DataFrame({
    "Crop": [le_crop.transform(["Rice"])[0]],
    "Season": [le_season.transform(["Kharif"])[0]],
    "Cost_of_Cultivation": [25000]
})

predicted_production = model.predict(sample_input)

print("\nPredicted Production for Sample Input:", predicted_production[0])
