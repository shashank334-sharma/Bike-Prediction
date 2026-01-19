import streamlit as st
import streamlit.components.v1 as components
import joblib
import numpy as np
import pandas as pd
import os

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Bike Demand Prediction",
    layout="wide"
)

# ======================================================
# CUSTOM CSS
# ======================================================
st.markdown("""
<style>
.main { background-color: #f5f7fa; }
.stButton>button {
    background-color: #4F8BF9;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 0.5em 2em;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# SESSION STATE DEFAULTS
# ======================================================
if "page" not in st.session_state:
    st.session_state.page = "🚲 Bike Demand Prediction"

# ======================================================
# SIDEBAR NAVIGATION
# ======================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🚲 Bike Demand Prediction", "📊 EDA Profile Report", "📈 Prediction Result"],
    index=["🚲 Bike Demand Prediction", "📊 EDA Profile Report", "📈 Prediction Result"].index(st.session_state.page)
)

# ======================================================
# 📊 EDA PROFILE REPORT PAGE
# ======================================================
if page == "📊 EDA Profile Report":
    st.title("📊 Bike Dataset – EDA Profile Report")

    html_file = "profile.html"
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            components.html(f.read(), height=1200, scrolling=True)
    else:
        st.error("❌ profile.html not found")

    st.stop()

# ======================================================
# 📈 PREDICTION RESULT PAGE
# ======================================================
if page == "📈 Prediction Result":
    st.title("📈 Bike Demand Prediction Result")

    if "prediction" not in st.session_state:
        st.warning("⚠️ No prediction found. Please predict first.")
    else:
        st.success(f"🚲 **Predicted Bike Demand:** {st.session_state.prediction:.0f}")

        st.subheader("🔍 Input Summary")
        st.dataframe(st.session_state.input_data, use_container_width=True)

    if st.button("🔙 Back to Prediction"):
        st.session_state.page = "🚲 Bike Demand Prediction"
        st.experimental_rerun()

    st.stop()

# ======================================================
# LOAD MODEL
# ======================================================
MODEL_PATH = "best_model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found")
    st.stop()

model = joblib.load(MODEL_PATH)

try:
    feature_names = model.feature_names_in_
except:
    feature_names = model.named_steps["preprocessor"].feature_names_in_

# ======================================================
# MAIN PREDICTION PAGE
# ======================================================
st.title("🚲 Bike Demand Prediction App")
st.write("Predict bike demand using manual input or CSV upload.")

SEASON_MAP = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}

option = st.radio("Choose input method:", ["Manual Input", "Upload CSV"])

# ======================================================
# MANUAL INPUT
# ======================================================
if option == "Manual Input":
    st.sidebar.header("Input Values")
    input_data = {}

    input_data["season"] = st.sidebar.selectbox("Season", list(SEASON_MAP.keys()))
    input_data["yr"] = st.sidebar.selectbox("Year", [2011, 2012])
    input_data["mnth"] = st.sidebar.slider("Month", 1, 12, 6)
    input_data["day"] = st.sidebar.slider("Day", 1, 31, 1)
    input_data["hr"] = st.sidebar.slider("Hour", 0, 23, 12)

    for flag in ["holiday", "weekday", "workingday", "is_peak_hour", "is_weekend"]:
        if flag in feature_names:
            input_data[flag] = st.sidebar.selectbox(flag.replace("_", " ").title(), [0, 1])

    if "atemp" in feature_names:
        input_data["atemp"] = st.sidebar.slider("Feels Like Temp", 0.0, 1.0, 0.5)

    if "windspeed" in feature_names:
        input_data["windspeed"] = st.sidebar.slider("Windspeed", 0.0, 1.0, 0.5)

    # Prepare model input
    row = {}
    for f in feature_names:
        if f == "season":
            row[f] = SEASON_MAP[input_data["season"]]
        elif f == "yr":
            row[f] = 1 if input_data["yr"] == 2012 else 0
        else:
            row[f] = input_data.get(f, 0)

    input_df = pd.DataFrame([row])[feature_names]

    pretty_df = input_df.copy()
    pretty_df.columns = [c.replace("_", " ").title() for c in pretty_df.columns]

    st.subheader("Selected Inputs")
    st.dataframe(pretty_df, use_container_width=True)

    if st.button("Predict"):
        prediction = model.predict(input_df)[0]

        # Save results
        st.session_state.prediction = prediction
        st.session_state.input_data = pretty_df
        st.session_state.page = "📈 Prediction Result"

        st.experimental_rerun()

# ======================================================
# CSV UPLOAD
# ======================================================
else:
    st.subheader("📁 Upload CSV File")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        if st.button("Predict for CSV"):
            for f in feature_names:
                if f not in df.columns:
                    df[f] = 0

            df = df[feature_names]
            preds = model.predict(df)

            st.success("✅ Prediction Completed")
            st.dataframe(pd.DataFrame({"Prediction": preds}))

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
---
<p style="text-align:center; color:#888;">Made with ❤️ using Streamlit</p>
""", unsafe_allow_html=True)
