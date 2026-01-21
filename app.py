import streamlit as st
import pandas as pd
import joblib

ODI_URL = "https://maic.qld.gov.au/wp-content/uploads/2016/02/Oswestry_Low_Back_Disability_Questionnaire.pdf"

st.set_page_config(page_title="Fall Risk Prediction", layout="centered")

# Load trained model
bundle = joblib.load("catboost_fall_model.joblib")
model = bundle["model"]
FEATURES = bundle["features"]

st.title("Fall risk predictor for older adults (>60 years old) with low back pain")
st.write("The predicted fall risk is generated using a trained machine learning model (CatBoost). "
    "This output is intended for **research and supportive screening purposes only**. "
    "Final clinical decisions should always be made by qualified healthcare professionals.")

st.subheader("Participant Inputs")

col1, col2 = st.columns(2)

with col1:
    gender_txt = st.selectbox("Gender", ["Female", "Male"], index=0)
    gender = 0 if gender_txt == "Female" else 1
    st.caption("Coding: Female = 0, Male = 1")

    height_cm = st.number_input("Height (cm)", min_value=80.0, max_value=250.0, value=170.0, step=0.1)

    hypertension = st.selectbox(
        "Hypertension (0 = No, 1 = Yes)",
        [0, 1],
        index=0,
        help="0 = No hypertension, 1 = Hypertension present."
    )

    medication_count = st.number_input(
        "Medication Count",
        min_value=0, max_value=50, value=0, step=1,
        help="Total number of regularly used medications."
    )

with col2:
    st.markdown("**ODI (Oswestry Disability Index)**")
    st.markdown(
        f"[Open ODI questionnaire PDF]({ODI_URL})  \n"
        "Please complete the questionnaire and enter the final ODI score (0–50) below."
    )
    odi = st.number_input("ODI Score (0–50)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

    st.markdown("**VAS (Visual Analogue Scale)**")
    vas = st.slider("Low back pain intensity (0 = none, 10 = worst imaginable)", 0.0, 10.0, 2.0, 0.1)

st.divider()
st.subheader("Postural Stability (PS) & Walking (W) Measures")
st.info(
    "📱 **Measurement Requirement**  \n"
    "For accurate model performance, **Postural Stability (PS)** and **Walking (W)** tests "
    "must be performed using the **Lockhart Monitor** application.  \n\n"
    "Please download **Lockhart Monitor** to your smartphone. "
    "Follow the in-app instructions to perform:  \n"
    "- **Postural Stability Test:** 30 seconds standing test  \n"
    "- **Walking Test:** 5-meter walking test  \n\n"
    "After completing the tests, enter the obtained values below."
)


col3, col4 = st.columns(2)

with col3:
    ps_velocity = st.number_input(
        "Postural Stability Velocity (cm/s)",
        value=0.0, step=0.01,
        help="PS = Postural Stability. Sway velocity during standing."
    )
    ps_sway_area = st.number_input(
        "Postural Stability Sway Area (cm²)",
        value=0.0, step=0.01,
        help="PS = Postural Stability. Sway area during standing."
    )
    ps_sway_path = st.number_input(
        "Postural Stability Sway Path (cm)",
        value=0.0, step=0.01,
        help="PS = Postural Stability. Total sway path length."
    )

with col4:
    w_velocity = st.number_input(
        "Walking Velocity (m/s)",
        value=0.0, step=0.01,
        help="W = Walking. Walking velocity."
    )
    w_duration = st.number_input(
        "Walking Duration (sec)",
        value=0.0, step=0.01,
        help="W = Walking. Duration of the walking task (e.g., seconds)."
    )

st.divider()
st.subheader("Prediction")



# Build input row in correct column order
input_dict = {
    "Gender": gender,
    "Height": height_cm,
    "Hypertension": hypertension,
    "ODI": odi,
    "VAS": vas,
    "PS_Velocity": ps_velocity,
    "PS_Sway_Area": ps_sway_area,
    "PS_Sway_Path": ps_sway_path,
    "W_Velocity": w_velocity,
    "W_Duration": w_duration,
    "Medication_Count": medication_count,
}

X_input = pd.DataFrame([input_dict], columns=FEATURES)

threshold = 0.50

if st.button("Predict"):

    # --- Input validation ---
    required_fields = {
        "Postural Stability Velocity": ps_velocity,
        "Postural Stability Sway Area": ps_sway_area,
        "Postural Stability Sway Path": ps_sway_path,
        "Walking Velocity": w_velocity,
        "Walking Duration": w_duration
    }

    missing = [name for name, val in required_fields.items() if val == 0]

    if len(missing) > 0:
        st.warning(
            "⚠️ Please complete the Postural Stability and Walking test measurements "
            "using the Lockhart Monitor app before prediction.\n\n"
            f"Missing or zero values: {', '.join(missing)}"
        )
        st.stop()

    # --- Prediction ---
    proba_fall = float(model.predict_proba(X_input)[0, 1])
    pred = 1 if proba_fall >= 0.50 else 0

    st.write(f"**Probability of Faller (%):** `{proba_fall:.3f}`")
    st.write(f"**Predicted Fall Status:** `{pred}`")

    if pred == 1:
        st.error("**High Fall Risk** (Predicted: Faller = 1)")
    else:
        st.success("**Low Fall Risk** (Predicted: Non-faller = 0)")