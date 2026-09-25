import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Heart Disease Prediction", page_icon="💓", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f7fafc 0%, #eef4f8 100%);
            color: #1f2937;
        }
        .main .block-container {
            padding-top: 3rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] span,
        label,
        .stMarkdown,
        .stMarkdown p {
            color: #0f172a !important;
        }
        [data-testid="stWidgetLabel"] {
            font-weight: 600 !important;
        }
        div[data-baseweb="input"],
        div[data-baseweb="select"] {
            background: #ffffff !important;
        }
        div[data-baseweb="input"] input {
            color: #0f172a !important;
            background: #ffffff !important;
        }
        div[data-baseweb="select"] > div {
            color: #0f172a !important;
            background: #ffffff !important;
        }
        .hero-card, .input-card, .result-card {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(255, 255, 255, 0.8);
            border-radius: 24px;
            box-shadow: 0 20px 50px rgba(15, 23, 42, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.65);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            padding: 1.25rem 1.4rem;
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.35rem;
            letter-spacing: -0.03em;
        }
        .hero-subtitle {
            color: #475569;
            font-size: 1rem;
            line-height: 1.5;
        }
        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.15rem;
        }
        .section-note {
            color: #64748b;
            margin-bottom: 0.9rem;
        }
        div[data-baseweb="select"], div[data-baseweb="input"] {
            border-radius: 16px !important;
        }
        .stButton > button {
            width: 100%;
            border: none;
            border-radius: 16px;
            padding: 0.85rem 1rem;
            font-weight: 700;
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            color: white;
            box-shadow: 0 12px 24px rgba(37, 99, 235, 0.24);
        }
        .stButton > button:hover {
            box-shadow: 0 16px 30px rgba(37, 99, 235, 0.32);
            transform: translateY(-1px);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the saved model and scaler
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Heart Disease Prediction</div>
        <div class="hero-subtitle">A clean, lightweight interface for entering patient details and getting a quick logistic-regression prediction.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="input-card"><div class="section-title">Basic details</div><div class="section-note">Core demographic information used by the model.</div>', unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
with right:
    sex = st.selectbox("Sex", options=["Male", "Female"])

basic_left, basic_right = st.columns(2)
with basic_left:
    chest_pain_type = st.selectbox("Chest Pain Type", options=["NAP", "ATA", "TA", "ASY"])
    RestingBP = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["Yes", "No"])
with basic_right:
    Cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    resting_ecg = st.selectbox("Resting ECG", options=["Normal", "ST", "LVH"])
    max_heart_rate = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="input-card"><div class="section-title">Clinical signals</div><div class="section-note">Exercise response and other measurements.</div>', unsafe_allow_html=True)
clinical_left, clinical_right = st.columns(2)
with clinical_left:
    exercise_induced_angina = st.selectbox("Exercise Induced Angina", options=["Yes", "No"])
with clinical_right:
    oldpeak = st.number_input("Oldpeak (ST depression induced by exercise)", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of the peak exercise ST segment", options=["Up", "Flat", "Down"])
st.markdown('</div>', unsafe_allow_html=True)

if st.button("Predict"):
    # Create a DataFrame for the input data
    input_data = {
        'age': [age],
        'sex': [1 if sex == "Male" else 0],
        'chest_pain_type': [chest_pain_type],
        'resting_blood_pressure': [RestingBP],
        'cholesterol': [Cholesterol],
        'fasting_blood_sugar': [1 if FastingBS == "Yes" else 0],
        'resting_ecg': [resting_ecg],
        'max_heart_rate': [max_heart_rate],
        'exercise_induced_angina': [1 if exercise_induced_angina == "Yes" else 0],
        'oldpeak': [oldpeak],
        'slope': [slope]
    }
    
    input_df = pd.DataFrame(input_data)
    scaler_input = pd.DataFrame({
        'Age': [age],
        'RestingBP': [RestingBP],
        'Cholesterol': [Cholesterol],
        'MaxHR': [max_heart_rate],
        'Oldpeak': [oldpeak]
    })
    input_df['RestingBP'] = RestingBP
    input_df['Cholesterol'] = Cholesterol
    input_df['FastingBS'] = 1 if FastingBS == "Yes" else 0
    
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df.reindex(columns=list(getattr(model, 'feature_names_in_', columns)), fill_value=0)
    scaled_values = scaler.transform(scaler_input)
    scaled_df = pd.DataFrame(scaled_values, columns=scaler_input.columns)
    input_df['Age'] = scaled_df['Age']
    input_df['RestingBP'] = scaled_df['RestingBP']
    input_df['Cholesterol'] = scaled_df['Cholesterol']
    input_df['MaxHR'] = scaled_df['MaxHR']
    input_df['Oldpeak'] = scaled_df['Oldpeak']
    input_df['Sex_M'] = 1 if sex == "Male" else 0

    if chest_pain_type == "ATA":
        input_df['ChestPainType_ATA'] = 1
    elif chest_pain_type == "NAP":
        input_df['ChestPainType_NAP'] = 1
    elif chest_pain_type == "TA":
        input_df['ChestPainType_TA'] = 1

    if resting_ecg == "Normal":
        input_df['RestingECG_Normal'] = 1
    elif resting_ecg == "ST":
        input_df['RestingECG_ST'] = 1

    if exercise_induced_angina == "Yes":
        input_df['ExerciseAngina_Y'] = 1

    if slope == "Flat":
        input_df['ST_Slope_Flat'] = 1
    elif slope == "Up":
        input_df['ST_Slope_Up'] = 1

    prediction = model.predict(input_df)[0]

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    if prediction == 1:
        st.error("The model predicts that you are likely to have heart disease. Please consult a healthcare professional.")
    else:
        st.success("The model predicts that you are unlikely to have heart disease. However, please consult a healthcare professional for a comprehensive evaluation.")
    st.markdown('</div>', unsafe_allow_html=True)

