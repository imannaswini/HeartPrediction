

````markdown
#  Heart Health Navigator: AI-Powered Heart Disease Prediction Web App

An interactive web application built with **Streamlit** that uses a **Random Forest** machine learning model to predict the risk of heart disease based on key medical attributes.

---

## 📝 Project Description

The **Heart Health Navigator** is a comprehensive, user-friendly tool designed to provide data-driven insights into cardiovascular health. It features:

- Secure **user authentication**.
- Educational sections with **interactive data visualizations**.
- An AI-powered **prediction tool**.
- A downloadable **summary report** for each prediction.

This project demonstrates a complete **end-to-end machine learning workflow**, from model training to deployment as an interactive web application.

---

## ✨ Key Features

- **User Authentication:** Secure Sign Up and Login system.
- **Multi-Page Interface:** Easy navigation via top menu bar.
- **Interactive Prediction Tool:** Input 13 key medical parameters for accurate risk assessment.
- **Data Visualization:** Explore heart disease datasets using interactive charts with Plotly.
- **Downloadable Reports:** Download prediction summaries as PDF reports.
- **Session History:** Keep a log of recent predictions and download as CSV.

---

## 💻 Tech Stack

- **Programming Language:** Python
- **Web Framework:** Streamlit
- **Machine Learning:** scikit-learn (Random Forest)
- **Data Manipulation:** pandas
- **Model Persistence:** joblib
- **Data Visualization:** plotly
- **PDF Generation:** FPDF2
- **UI Components:** streamlit-option-menu

---

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

- Python 3.9+
- pip (Python package installer)

### Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/heart-health-navigator.git
cd heart-health-navigator
````

2. **Install dependencies:**
   It's recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

3. **Ensure the following files are present in the main project directory:**

   * `heart.csv` (dataset)
   * `heart_disease_model.joblib` (trained model)
   * `scaler.joblib` (trained scaler)

4. **User Database:**
   On the first run, the app will automatically create a `users.csv` file:

```csv
username,password,email
```

### Running the Application

```bash
streamlit run app.py
```

The app will open in your default web browser.

---

## 📁 Project Structure

```
.
├── .streamlit/
│   └── config.toml
├── app.py
├── heart.csv
├── heart_disease_model.joblib
├── scaler.joblib
├── users.csv
└── requirements.txt
```

---

## 👤 Author

**Mannaswini P A**

* 📧 Email: [iammannaswini@gmail.com](mailto:iammannaswini@gmail.com)
* LinkedIn: [linkedin.com/in/mannaswini-p-a](https://www.linkedin.com/in/mannaswini-p-a-7s7r7)
* GitHub: [@imannaswini](https://github.com/imannaswini)

---

## 📝 `requirements.txt`

```text
streamlit
pandas
scikit-learn
joblib
plotly
fpdf2
streamlit-option-menu
```

---

```



```
