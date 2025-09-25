import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os
from datetime import datetime
from fpdf import FPDF
from streamlit_option_menu import option_menu

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Heart Health Navigator",
    page_icon="❤️",
    layout="wide"
)

# --- 2. User Authentication and Data Handling ---
if not os.path.exists('users.csv'):
    pd.DataFrame(columns=['username', 'password', 'email']).to_csv('users.csv', index=False)

def load_user_data():
    return pd.read_csv('users.csv')

def save_user_data(df):
    df.to_csv('users.csv', index=False)

@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('heart_disease_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler files not found. Ensure they are in the project directory.")
        return None, None

@st.cache_data
def load_data():
    try:
        return pd.read_csv('heart.csv')
    except FileNotFoundError:
        return None

model, scaler = load_model_and_scaler()
df = load_data()

# --- PDF Report Generation ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Heart Health Risk Assessment Report', 0, 1, 'C')
        self.ln(10)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

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
    
    return bytes(pdf.output(dest='S'))

# --- 3. Page Definitions ---

def show_login_page():
    st.title("Welcome to the Heart Health Navigator")
    st.write("Please log in or sign up to continue.")
    tabs = st.tabs(["Login", "Sign Up"])
    with tabs[0]:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                users_df = load_user_data()
                user = users_df[(users_df['username'] == username) & (users_df['password'] == password)]
                if not user.empty:
                    st.session_state.logged_in = True
                    st.session_state.username = user.iloc[0]['username']
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")
    with tabs[1]:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a Username")
            email = st.text_input("Your Email Address")
            new_password = st.text_input("Choose a Password", type="password")
            signup_submitted = st.form_submit_button("Sign Up")
            if signup_submitted:
                users_df = load_user_data()
                if new_username in users_df['username'].values:
                    st.error("Username already exists.")
                else:
                    new_user = pd.DataFrame([[new_username, new_password, email]], columns=['username', 'password', 'email'])
                    updated_users_df = pd.concat([users_df, new_user], ignore_index=True)
                    save_user_data(updated_users_df)
                    st.success("Account created! Please proceed to the Login tab.")

def show_homepage():
    st.title(f"Welcome to the Navigator, {st.session_state.username}! ❤️")
    st.markdown("---")
    st.image("https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?q=80&w=2070&auto.format&fit=crop", caption="Leveraging AI for a Healthier Tomorrow", use_column_width=True)
    st.markdown("## Your Guide to Understanding Cardiovascular Health")
    st.markdown("This application provides a suite of tools to help you learn about heart disease and assess potential risks using a powerful machine learning model. Explore the sections using the navigation bar above.")

def show_info_page():
    st.title("Comprehensive Guide to Heart Disease 🩺")
    st.markdown("---")
    st.subheader("What is Heart Disease?")
    st.write("Heart disease refers to a range of conditions that affect your heart, including blood vessel diseases, arrhythmias, and congenital heart defects.")

    st.subheader("Causes, Symptoms, and Precautions")
    tab1, tab2, tab3 = st.tabs(["🚨 Common Symptoms", "🛡️ Risk Factors & Precautions", "🌍 Global Impact"])
    with tab1:
        st.write("- **Chest Pain (Angina):** A feeling of pressure or tightness in the chest.")
        st.write("- **Shortness of Breath:** Difficulty breathing, especially with activity.")
        st.write("- **Fatigue:** Unexplained weakness or tiredness.")
    with tab2:
        st.write("- **Key Risk Factors:** High blood pressure, high cholesterol, smoking, diabetes, and obesity.")
        st.write("- **Preventive Measures:** Focus on a healthy diet, regular exercise, and avoiding smoking.")
    with tab3:
        st.write("Cardiovascular diseases are the leading cause of death globally, but many forms are preventable through awareness and lifestyle changes.")

    st.subheader("Food Intake Suggestions by Age")
    with st.expander("Click to see dietary recommendations"):
        st.markdown("""
        - **20-30s:** Focus on building healthy habits. Incorporate lean proteins, a variety of vegetables, and whole grains.
        - **40-50s:** Pay attention to sodium intake to manage blood pressure. Increase intake of omega-3 fatty acids (e.g., salmon, walnuts).
        - **60s and beyond:** Ensure adequate calcium and Vitamin D for bone and heart health. Focus on fiber-rich foods to manage cholesterol.
        """)

def show_prediction_page(model, scaler):
    st.title("AI-Powered Risk Prediction Tool")
    st.markdown("This tool uses a Random Forest model to assess risk based on 13 key clinical features. Please fill in the patient's details accurately for the most reliable prediction.")
    st.markdown("---")
    if 'history' not in st.session_state:
        st.session_state.history = []
    with st.form("prediction_form"):
        st.subheader("Patient Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input('Age', 1, 120, 52)
            sex = st.selectbox('Sex', ('Male', 'Female'))
            cp = st.selectbox('Chest Pain Type', (0, 1, 2, 3))
            trestbps = st.number_input('Resting Blood Pressure (mm Hg)', 80, 220, 120)
        with col2:
            chol = st.slider('Serum Cholesterol (mg/dl)', 100, 600, 200)
            fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ('False', 'True'))
            restecg = st.selectbox('Resting ECG Results', (0, 1, 2))
            thalach = st.slider('Maximum Heart Rate Achieved', 60, 220, 150)
        with col3:
            exang = st.selectbox('Exercise Induced Angina', ('No', 'Yes'))
            oldpeak = st.slider('ST Depression (Oldpeak)', 0.0, 7.0, 1.0, 0.1)
            slope = st.selectbox('Slope of Peak Exercise ST Segment', (0, 1, 2))
            ca = st.selectbox('Number of Major Vessels Colored', (0, 1, 2, 3, 4))
            thal = st.selectbox('Thalassemia', (0, 1, 2, 3))
        
        submit_button = st.form_submit_button(label='**Analyze Patient Risk**', use_container_width=True, type="primary")

    if submit_button:
        with st.spinner('Analyzing data...'):
            user_input_data = {'Age': age, 'Sex': sex, 'Chest Pain Type': cp, 'Cholesterol': chol}
            model_input_data = {'age': age, 'sex': 1 if sex == 'Male' else 0, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': 1 if fbs == 'True' else 0, 'restecg': restecg, 'thalach': thalach, 'exang': 1 if exang == 'Yes' else 0, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal}
            input_df = pd.DataFrame([model_input_data])
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            result_text = "High Risk" if prediction == 1 else "Low Risk"
            confidence = prediction_proba[prediction] * 100
            
            st.session_state.history.insert(0, {'Date': datetime.now().strftime('%Y-%m-%d %H:%M'), 'Prediction': result_text, 'Confidence': f"{confidence:.2f}%", **user_input_data})
            
            st.markdown("---")
            st.header("Analysis Complete")
            
            # --- FIX APPLIED HERE: SHAP PLOT REMOVED ---
            # The results are now in two tabs instead of three.
            tab1, tab2 = st.tabs(["📊 Summary", "📋 History Log"])
            with tab1:
                if prediction == 1:
                    st.warning(f'**Overall Assessment:** The model predicts a **{result_text}** of Heart Disease.')
                else:
                    st.success(f'**Overall Assessment:** The model predicts a **{result_text}** of Heart Disease.')
                st.metric("Model Confidence Score", f"{confidence:.2f}%")
                pdf_data = create_pdf_report(user_input_data, result_text, confidence)
                st.download_button(label="📄 Download Full Report (PDF)", data=pdf_data, file_name="Heart_Health_Report.pdf", mime="application/pdf")

            with tab2:
                st.subheader("Recent Prediction Log")
                if st.session_state.history:
                    history_df = pd.DataFrame(st.session_state.history)
                    st.dataframe(history_df, use_container_width=True)
                    csv_history = history_df.to_csv(index=False).encode('utf-8')
                    st.download_button(label="📥 Download Log as CSV", data=csv_history, file_name='prediction_log.csv', mime='text/csv')

def show_about_contact_page():
    st.title("About This Project & Contact Us")
    st.markdown("---")
    st.subheader("About the Project")
    st.write("This application is a B.Tech mini-project designed to demonstrate a complete machine learning workflow, from model training to deployment as an interactive web application.")
    st.subheader("Future Improvements")
    st.write("- Integration with a live database for user data storage.")
    st.write("- Adding more advanced models and allowing users to compare their predictions.")
    st.markdown("---")
    st.subheader("Contact Information")
    st.write("This project was created by **Mannaswini P A**.")
    st.markdown("- 📧 **Email:** [iammannaswini@gmail.com](mailto:iammannaswini@gmail.com)\n- 👔 **LinkedIn:** [linkedin.com/in/mannaswini-p-a](https://www.linkedin.com/in/mannaswini-p-a-7s7r7)\n- 💻 **GitHub:** [github.com/imannaswini](https://github.com/imannaswini)")

# --- 4. Main App Logic ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

if not st.session_state.logged_in:
    show_login_page()
else:
    selected = option_menu(
        menu_title=None,
        options=["Home", "Heart Info", "Prediction Tool", "About & Contact Us"],
        icons=['house', 'heart-pulse', 'clipboard2-data', 'person-lines-fill'],
        default_index=0,
        orientation="horizontal",
    )
    with st.sidebar:
        st.title(f"Welcome, {st.session_state.username}!")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()
            
    if selected == "Home":
        show_homepage()
    elif selected == "Heart Info":
        show_info_page()
    elif selected == "Prediction Tool":
        show_prediction_page(model, scaler)
    elif selected == "About & Contact Us":
        show_about_contact_page()