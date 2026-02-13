import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Sample Dataset
# -----------------------------
data = {
    "Crop": ["Rice", "Wheat", "Maize", "Rice", "Wheat", "Maize", "Rice", "Wheat"],
    "Season": ["Kharif", "Rabi", "Kharif", "Rabi", "Kharif", "Rabi", "Kharif", "Rabi"],
    "Cost_of_Cultivation": [20000, 18000, 15000, 21000, 17000, 16000, 22000, 17500],
    "Production": [50, 45, 40, 55, 43, 42, 60, 44]
}

df = pd.DataFrame(data)

# Encode categorical data
le_crop = LabelEncoder()
le_season = LabelEncoder()

df["Crop"] = le_crop.fit_transform(df["Crop"])
df["Season"] = le_season.fit_transform(df["Season"])

# Split
X = df[["Crop", "Season", "Cost_of_Cultivation"]]
y = df["Production"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n===== MODEL PERFORMANCE =====")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")
print("=============================\n")

# -----------------------------
# Take User Input
# -----------------------------
print("Enter Crop Type (Rice/Wheat/Maize):")
crop_input = input()

print("Enter Season (Kharif/Rabi):")
season_input = input()

print("Enter Cost of Cultivation:")
cost_input = float(input())

# Convert input
crop_encoded = le_crop.transform([crop_input])[0]
season_encoded = le_season.transform([season_input])[0]

sample = pd.DataFrame({
    "Crop": [crop_encoded],
    "Season": [season_encoded],
    "Cost_of_Cultivation": [cost_input]
})

prediction = model.predict(sample)

print("\n===== PREDICTION RESULT =====")
print(f"Predicted Crop Production: {prediction[0]:.2f} units")
print("=============================")
