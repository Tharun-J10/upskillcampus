import streamlit as st
import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Agri Analytics Dashboard", layout="wide")

# -------------------------------------------------
# CUSTOM CSS STYLE
# -------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #f1f8e9, #ffffff);
}
.header {
    font-size:38px;
    font-weight:bold;
    color:#1b5e20;
}
.subheader {
    font-size:18px;
    color:#2e7d32;
}
.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    text-align: center;
}
.footer {
    position: fixed;
    bottom: 10px;
    right: 20px;
    font-size: 14px;
    color: grey;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOGIN SYSTEM
# -------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Agricultural Analytics Login")
    st.markdown("### Developed by Tharun J BE ECE")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "1234":
            st.session_state.logged_in = True
        else:
            st.error("Invalid Credentials")

    st.stop()

# -------------------------------------------------
# HEADER WITH LOGO
# -------------------------------------------------
col_logo, col_title = st.columns([1,5])

with col_logo:
    st.image("logo.png", width=90)  # Make sure logo.png exists in folder

with col_title:
    st.markdown("<div class='header'>🌾 Government Crop Production Analytics Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>AI Powered Agricultural Decision Support System</div>", unsafe_allow_html=True)
    st.markdown("#### Developed by Tharun J BE ECE")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
df = pd.read_csv("crop_data.csv")

le_crop = LabelEncoder()
le_season = LabelEncoder()

df["Crop"] = le_crop.fit_transform(df["Crop"])
df["Season"] = le_season.fit_transform(df["Season"])

X = df[["Crop", "Season", "Cost_of_Cultivation"]]
y = df["Production"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

pred_test = model.predict(X_test)
r2 = r2_score(y_test, pred_test)

# -------------------------------------------------
# SIDEBAR INPUTS
# -------------------------------------------------
st.sidebar.markdown("## 📊 Input Parameters")
st.sidebar.markdown("### Developed by Tharun J BE ECE")

state = st.sidebar.selectbox("Select State", df["State"].unique())
districts = df[df["State"] == state]["District"].unique()
district = st.sidebar.selectbox("Select District", districts)

crop_name = st.sidebar.selectbox("Select Crop", le_crop.classes_)
season_name = st.sidebar.selectbox("Select Season", le_season.classes_)

cost = st.sidebar.number_input("Cost of Cultivation", min_value=1000, max_value=100000, step=1000)

st.sidebar.metric("Model Accuracy (R²)", f"{r2:.2f}")

predict = st.sidebar.button("🚀 Predict")

# -------------------------------------------------
# PREDICTION SECTION
# -------------------------------------------------
if predict:

    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    crop_encoded = le_crop.transform([crop_name])[0]
    season_encoded = le_season.transform([season_name])[0]

    sample = pd.DataFrame({
        "Crop": [crop_encoded],
        "Season": [season_encoded],
        "Cost_of_Cultivation": [cost]
    })

    prediction = model.predict(sample)[0]

    st.success(f"🌱 Estimated Production: {prediction:.2f} units")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class='card'>
            <h4>💰 Cost</h4>
            <h2>{cost}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='card'>
            <h4>🌾 Predicted Yield</h4>
            <h2>{prediction:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='card'>
            <h4>📊 Model Accuracy</h4>
            <h2>{r2:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    # Chart
    fig, ax = plt.subplots()
    ax.bar(["Cost (x1000)", "Predicted Yield"], [cost/1000, prediction])
    ax.set_title("Cost vs Yield Comparison")
    st.pyplot(fig)

    # CSV Download
    result_df = pd.DataFrame({
        "State": [state],
        "District": [district],
        "Crop": [crop_name],
        "Season": [season_name],
        "Cost_of_Cultivation": [cost],
        "Predicted_Production": [prediction]
    })

    csv = result_df.to_csv(index=False)

    st.download_button(
        "⬇ Download Prediction Report",
        csv,
        "prediction_report.csv",
        "text/csv"
    )

# -------------------------------------------------
# ANALYTICS SECTION
# -------------------------------------------------
st.markdown("## 📈 Historical Production Analytics")

colA, colB = st.columns(2)

with colA:
    crop_trend = df.groupby("Crop")["Production"].mean()
    st.line_chart(crop_trend)

with colB:
    state_trend = df.groupby("State")["Production"].mean()
    st.bar_chart(state_trend)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("<div class='footer'>Developed by Tharun J BE ECE</div>", unsafe_allow_html=True)
