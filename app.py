import streamlit as st
import joblib
import numpy as np
# This is the explicit, correct fix
from fpdf2 import FPDF
from datetime import datetime

# ---------------- PDF CLASS ----------------
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Heart Disease Prediction Report', 0, 1, 'C')
        self.ln(5)

# ---------------- PDF CREATION ----------------
def create_pdf_report(input_data, result_text, confidence):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Prediction Summary', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(50, 10, 'Overall Risk:', 0, 0)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"{result_text} (Confidence: {confidence:.2f}%)", 0, 1)
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Patient Data Provided', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    for key, value in input_data.items():
        pdf.cell(50, 10, f"{key.replace('_', ' ').title()}:", 0, 0)
        pdf.cell(0, 10, str(value), 0, 1)
    
    # FIX: return bytes directly for Streamlit
    return pdf.output(dest='S')  # returns bytes

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model():
    model = joblib.load('heart_disease_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

# ---------------- PREDICTION PAGE ----------------
def show_prediction_page(model, scaler):
    st.title("Heart Disease Prediction Tool")

    # ===================== USER GUIDELINES =====================
    with st.expander(" User Guidelines"):
        st.markdown("""
         Welcome to the Heart Disease Prediction Tool!
        This AI-powered tool estimates your **risk of heart disease** based on your health data.

        How It Works
        - Uses a trained **Machine Learning model** based on real patient datasets.
        - Predicts your **heart disease risk** and provides a **confidence score**.

        How to Use
        1. Enter all health parameters accurately.
        2. Click **"🔍 Predict"** to view your result.
        3. Download your personalized **PDF Report** for reference.

        Disclaimer
        - This tool is for **educational and awareness purposes only**.
        - It is **not a medical diagnosis**.
        - Always consult a **healthcare professional** for advice.
        """)

    # ===================== TOOL INFO =====================
    with st.expander("Model & Accuracy Information"):
        st.markdown("""
        ### 🔧 Model Details
        - **Algorithm:** Logistic Regression  
        - **Dataset:** UCI Heart Disease Dataset  
        - **Model Accuracy:** ~85% on validation data  
        - **Framework:** Scikit-learn 1.3+  
        - **Normalization:** StandardScaler applied  

        ### Prediction Meaning
        - **At Risk:** Higher probability of heart disease.  
        - **Low Risk:** Lower probability.  
        - **Confidence Score:** Indicates model certainty.
        """)

    # ===================== TERM DETAILS =====================
    with st.expander("Input Terms Explained"):
        st.markdown("""
        | Term | Description |
        |------|--------------|
        | **Age** | Age in years |
        | **Sex** | 0 = Female, 1 = Male |
        | **cp (Chest Pain Type)** | 0: Typical Angina, 1: Atypical, 2: Non-anginal, 3: Asymptomatic |
        | **trestbps** | Resting Blood Pressure (mm Hg) |
        | **chol** | Serum Cholesterol (mg/dl) |
        | **fbs** | Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False) |
        | **restecg** | Resting ECG Results (0 = Normal, 1 = ST-T abnormality, 2 = LV Hypertrophy) |
        | **thalach** | Maximum Heart Rate Achieved |
        | **exang** | Exercise-Induced Angina (1 = Yes, 0 = No) |
        | **oldpeak** | ST Depression Induced by Exercise |
        | **slope** | Slope of Peak Exercise ST Segment (0–2) |
        | **ca** | Major Vessels Colored by Fluoroscopy (0–4) |
        | **thal** | Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect) |
        """)

    st.markdown("---")
    st.header("Enter Your Health Details")

    # ---------------- INPUT SECTION ----------------
    age = st.number_input("Age", 1, 120, 30)
    sex = st.selectbox("Sex (0: Female, 1: Male)", [0, 1])
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1: Yes, 0: No)", [0, 1])
    restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina (1: Yes, 0: No)", [0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (0–2)", [0, 1, 2])
    ca = st.selectbox("Major Vessels Colored by Fluoroscopy (0–4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (1–3)", [1, 2, 3])

    user_input = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }

    input_array = np.array(list(user_input.values())).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    # ---------------- PREDICTION SECTION ----------------
    if st.button(" Predict"):
        prediction = model.predict(input_scaled)[0]
        confidence = model.predict_proba(input_scaled)[0][int(prediction)] * 100
        result_text = "At Risk" if prediction == 1 else "Low Risk"

        st.success(f"Prediction: **{result_text}** (Confidence: {confidence:.2f}%)")

        pdf_data = create_pdf_report(user_input, result_text, confidence)
        st.download_button(
            label=" Download Full Report (PDF)",
            data=pdf_data,
            file_name="Heart_Health_Report.pdf",
            mime="application/pdf"
        )

# ---------------- HOMEPAGE ----------------
def show_home_page():
    st.title("Welcome to the Heart Health Prediction Platform")

    st.markdown("""
     About This Platform
    This interactive web app allows you to estimate your **risk of heart disease**
    based on standard medical health indicators.

     Features:
    - AI-based **Heart Disease Risk Prediction**
    - **Confidence Score** for predictions  
    - Downloadable **PDF Report**
    - Explanation of all input medical terms  

     How It Works:
    1. The model was trained on the **UCI Heart Disease dataset**.
    2. It uses a **Logistic Regression algorithm** for binary classification.
    3. Input features are **standardized using StandardScaler**.
    4. The model outputs a **risk prediction** with probability confidence.

    Disclaimer:
    - This is **not a diagnostic tool**.
    - Always consult a **medical professional** for clinical advice.
    """)

    st.info(" Use the sidebar to navigate to the **Prediction Tool**.")

# ---------------- MAIN ----------------
def main():
    st.sidebar.title("🔍 Navigation")
    page = st.sidebar.radio("Go to:", ["Home", " Prediction Tool"])

    model, scaler = load_model()

    if page == "Home":
        show_home_page()
    elif page == " Prediction Tool":
        show_prediction_page(model, scaler)

if __name__ == "__main__":
    main()
