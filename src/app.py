import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import os
from fpdf import FPDF
import datetime
import pandas as pd

def generate_pdf(patient_name, results):
    file_path = "patient_report.pdf"

    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "AI Medical Report", ln=True, align="C")

    pdf.ln(5)

    # Patient info
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 8, f"Patient Name: {patient_name}", ln=True)
    pdf.cell(200, 8, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)

    pdf.ln(5)

    # Results
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Prediction Summary:", ln=True)

    pdf.set_font("Arial", "", 12)

    for disease, prob in results.items():
        if prob > 0.6:
            status = "High Risk"
        elif prob > 0.3:
            status = "Moderate Risk"
        else:
            status = "Low Risk"

        pdf.cell(200, 8, f"{disease}: {status} ({prob:.2f})", ln=True)

    pdf.ln(5)

    # Recommendation
    max_disease = max(results, key=results.get)
    max_value = results[max_disease]

    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Doctor Recommendation:", ln=True)

    pdf.set_font("Arial", "", 12)

    if max_value > 0.6:
        pdf.multi_cell(0, 8, f"Immediate attention required for {max_disease}.")
    elif max_value > 0.3:
        pdf.multi_cell(0, 8, f"Further tests recommended for {max_disease}.")
    else:
        pdf.multi_cell(0, 8, "Patient condition appears stable.")

    pdf.output(file_path)

    return file_path


def generate_heart_report(
    patient_name,
    inputs,
    prediction,
    probability,
    recommendations
):

    file_path = "heart_report.pdf"

    pdf = FPDF()

    pdf.add_page()

    # =====================================
    # TITLE
    # =====================================

    pdf.set_font("Arial", "B", 18)

    pdf.cell(
        200,
        10,
        "AI Heart Disease Report",
        ln=True,
        align="C"
    )

    pdf.ln(5)

    # =====================================
    # PATIENT INFO
    # =====================================

    pdf.set_font("Arial", "", 12)

    pdf.cell(
        200,
        8,
        f"Patient Name: {patient_name}",
        ln=True
    )

    pdf.cell(
        200,
        8,
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ln=True
    )

    pdf.ln(5)

    # =====================================
    # INPUT PARAMETERS
    # =====================================

    pdf.set_font("Arial", "B", 14)

    pdf.cell(
        200,
        10,
        "Patient Parameters",
        ln=True
    )

    pdf.set_font("Arial", "", 12)

    for key, value in inputs.items():

        pdf.cell(
            200,
            8,
            f"{key}: {value}",
            ln=True
        )

    pdf.ln(5)

    # =====================================
    # PREDICTION RESULT
    # =====================================

    pdf.set_font("Arial", "B", 14)

    pdf.cell(
        200,
        10,
        "Prediction Result",
        ln=True
    )

    pdf.set_font("Arial", "", 12)

    if prediction == 1:

        pdf.multi_cell(
            0,
            8,
            f"Heart Disease Detected\n"
            f"Risk Probability: {probability*100:.2f}%"
        )

    else:

        pdf.multi_cell(
            0,
            8,
            f"No Heart Disease Detected\n"
            f"Confidence: {(1-probability)*100:.2f}%"
        )

    pdf.ln(5)

    # =====================================
    # CLINICAL ANALYSIS
    # =====================================

    pdf.set_font("Arial", "B", 14)

    pdf.cell(
        200,
        10,
        "Clinical Analysis",
        ln=True
    )

    pdf.set_font("Arial", "", 12)

    if probability > 0.75:

        pdf.multi_cell(
            0,
            8,
            "HIGH CARDIAC RISK DETECTED.\n"
            "Immediate cardiology consultation recommended."
        )

    elif probability > 0.45:

        pdf.multi_cell(
            0,
            8,
            "MODERATE CARDIAC RISK DETECTED.\n"
            "Lifestyle modification advised."
        )

    else:

        pdf.multi_cell(
            0,
            8,
            "LOW CARDIAC RISK DETECTED.\n"
            "Maintain healthy lifestyle."
        )

    pdf.ln(5)

    # =====================================
    # AI RECOMMENDATIONS
    # =====================================

    pdf.set_font("Arial", "B", 14)

    pdf.cell(
        200,
        10,
        "AI Medical Recommendations",
        ln=True
    )

    pdf.set_font("Arial", "", 12)

    if recommendations:

        for rec in recommendations:

            pdf.multi_cell(
                0,
                8,
                f"- {rec}"
            )

    else:

        pdf.multi_cell(
            0,
            8,
            "No major medical risks detected."
        )

    pdf.ln(5)

    # =====================================
    # HEART HEALTH PLAN
    # =====================================

    pdf.set_font("Arial", "B", 14)

    pdf.cell(
        200,
        10,
        "Heart Healthy Lifestyle",
        ln=True
    )

    pdf.set_font("Arial", "", 12)

    lifestyle = """
Recommended:
- Daily walking
- Low salt diet
- Fruits and vegetables
- Proper sleep
- Stress reduction

Avoid:
- Smoking
- Alcohol
- Fried foods
- Excess sugar
- Excess oily foods
"""

    pdf.multi_cell(
        0,
        8,
        lifestyle
    )

    pdf.ln(5)

    # =====================================
    # DISCLAIMER
    # =====================================

    pdf.set_font("Arial", "I", 10)

    pdf.multi_cell(
        0,
        6,
        "This AI-generated report should not replace professional medical diagnosis."
    )

    pdf.output(file_path)

    return file_path


def generate_pdf(patient_name, results, recommendations=None):

    file_path = "patient_report.pdf"

    pdf = FPDF()

    pdf.add_page()

    # ==========================================
    # TITLE
    # ==========================================

    pdf.set_font("Arial", "B", 18)

    pdf.cell(
        200,
        10,
        "AI Medical Report",
        ln=True,
        align="C"
    )

    pdf.ln(5)

    # ==========================================
    # PATIENT INFORMATION
    # ==========================================

    pdf.set_font("Arial", "", 12)

    pdf.cell(
        200,
        8,
        f"Patient Name: {patient_name}",
        ln=True
    )

    pdf.cell(
        200,
        8,
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ln=True
    )

    pdf.ln(5)

    # ==========================================
    # PREDICTION SUMMARY
    # ==========================================

    pdf.set_font("Arial", "B", 14)

    pdf.cell(
        200,
        10,
        "Prediction Summary",
        ln=True
    )

    pdf.set_font("Arial", "", 12)

    for disease, prob in results.items():

        if prob > 0.7:
            status = "HIGH RISK"

        elif prob > 0.4:
            status = "MODERATE RISK"

        else:
            status = "LOW RISK"

        pdf.cell(
            200,
            8,
            f"{disease}: {status} ({prob*100:.2f}%)",
            ln=True
        )

    pdf.ln(5)

    # ==========================================
    # CLINICAL INSIGHT
    # ==========================================

    max_disease = max(results, key=results.get)

    max_value = results[max_disease]

    pdf.set_font("Arial", "B", 14)

    pdf.cell(
        200,
        10,
        "Clinical Insight",
        ln=True
    )

    pdf.set_font("Arial", "", 12)

    if max_value > 0.7:

        pdf.multi_cell(
            0,
            8,
            f"High risk detected for {max_disease}. "
            f"Immediate medical consultation recommended."
        )

    elif max_value > 0.4:

        pdf.multi_cell(
            0,
            8,
            f"Moderate risk detected for {max_disease}. "
            f"Lifestyle modification and monitoring advised."
        )

    else:

        pdf.multi_cell(
            0,
            8,
            "Patient condition appears stable."
        )

    pdf.ln(5)

    # ==========================================
    # AI RECOMMENDATIONS
    # ==========================================

    pdf.set_font("Arial", "B", 14)

    pdf.cell(
        200,
        10,
        "AI Medical Recommendations",
        ln=True
    )

    pdf.set_font("Arial", "", 12)

    if recommendations and len(recommendations) > 0:

        for rec in recommendations:

            pdf.multi_cell(
                0,
                8,
                f"- {rec}"
            )

    else:

        pdf.multi_cell(
            0,
            8,
            "No major medical risks detected."
        )

    pdf.ln(5)

    # ==========================================
    # HEALTHY LIFESTYLE
    # ==========================================

    pdf.set_font("Arial", "B", 14)

    pdf.cell(
        200,
        10,
        "Recommended Healthy Lifestyle",
        ln=True
    )

    pdf.set_font("Arial", "", 12)

    lifestyle = """
Recommended:
- Fruits and vegetables
- Daily walking
- Regular exercise
- Proper hydration
- Good sleep
- Stress reduction

Avoid:
- Smoking
- Alcohol
- Fried foods
- Sugary drinks
- Excess salt
- Junk foods
"""

    pdf.multi_cell(
        0,
        8,
        lifestyle
    )

    pdf.ln(5)

    # ==========================================
    # DISCLAIMER
    # ==========================================

    pdf.set_font("Arial", "I", 10)

    pdf.multi_cell(
        0,
        6,
        "This report is AI-generated and should not replace professional medical diagnosis."
    )

    # ==========================================
    # SAVE PDF
    # ==========================================

    pdf.output(file_path)

    return file_path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Ensure page config is set before any other Streamlit command
st.set_page_config(page_title="Multiple Disease Prediction System", layout="centered")

#Heart Disease
heart_disease_model = joblib.load(
    os.path.join(BASE_DIR, "models", "heart_xgb_model.pkl")
)

heart_scaler = joblib.load(
    os.path.join(BASE_DIR, "models", "heart_scaler.pkl")
)

heart_imputer = joblib.load(
    os.path.join(BASE_DIR, "models", "heart_imputer.pkl")
)

# Breast Cancer (Tabular ML)
breast_model = joblib.load(os.path.join(BASE_DIR, "models", "breast_model.pkl"))
breast_scaler = joblib.load(os.path.join(BASE_DIR, "models", "breast_scaler.pkl"))
breast_features = joblib.load(os.path.join(BASE_DIR, "models", "breast_features.pkl"))


lung_model = joblib.load(os.path.join(BASE_DIR, "models", "lung_model.pkl"))
lung_scaler = joblib.load(os.path.join(BASE_DIR, "models", "lung_scaler.pkl"))
lung_features = joblib.load(os.path.join(BASE_DIR, "models", "lung_features.pkl"))

diabetes_model = pickle.load(open(r'C:\Users\vivek\OneDrive\Desktop\New folder\Multiple-Disease-Prediction-System\models\diabetes_model.sav', 'rb'))

parkinsons_model = pickle.load(open(r'C:\Users\vivek\OneDrive\Desktop\New folder\Multiple-Disease-Prediction-System\models\parkinsons_model.sav', 'rb'))

kidney_model = pickle.load(open(r'C:\Users\vivek\OneDrive\Desktop\New folder\Multiple-Disease-Prediction-System\models\kidney_rf_model.sav', 'rb'))



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "preprocessing", "stroke_model.pkl")
scaler_path = os.path.join(BASE_DIR, "preprocessing", "stroke_scaler.pkl")
encoder_path = os.path.join(BASE_DIR, "preprocessing", "stroke_encoders.pkl")

stroke_model = joblib.load(model_path)
stroke_scaler = joblib.load(scaler_path)
stroke_encoders = joblib.load(encoder_path)


@st.cache_resource
def load_kidney_artifacts():
    import pickle

    model = pickle.load(open(r'C:\Users\vivek\OneDrive\Desktop\New folder\Multiple-Disease-Prediction-System\models\kidney_rf_model.sav', 'rb'))

    with open(r'C:\Users\vivek\OneDrive\Desktop\New folder\Multiple-Disease-Prediction-System\preprocessing\columns.pkl', 'rb') as f:
        features = pickle.load(f)

    with open(r'C:\Users\vivek\OneDrive\Desktop\New folder\Multiple-Disease-Prediction-System\preprocessing\scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open(r'C:\Users\vivek\OneDrive\Desktop\New folder\Multiple-Disease-Prediction-System\preprocessing\encoder.pkl', 'rb') as f:
        encoders = pickle.load(f)

    return model, features, scaler, encoders


@st.cache_resource
def load_eye_model():
    import tensorflow as tf

    # Compatibility wrapper: older/newer Keras saved the Resizing layer
    # with an 'antialias' argument that some runtime versions don't accept.
    class ResizingCompat(tf.keras.layers.Resizing):
        @classmethod
        def from_config(cls, config):
            config = dict(config)
            config.pop('antialias', None)
            return super().from_config(config)

    # Compatibility wrapper for Dense layers that may include
    # a 'quantization_config' key in older/newer saved configs.
    class DenseCompat(tf.keras.layers.Dense):
        @classmethod
        def from_config(cls, config):
            config = dict(config)
            config.pop('quantization_config', None)
            return super().from_config(config)

    custom_objects = {"Resizing": ResizingCompat, "Dense": DenseCompat}

    return tf.keras.models.load_model(
        r'C:\Users\vivek\OneDrive\Desktop\New folder\Multiple-Disease-Prediction-System\models\Eye.h5',
        compile=False,
        custom_objects=custom_objects
    )


# If the query param `page=chat` is present, load the chat app in this tab
params = st.query_params
if params.get("page", [""])[0] == "chat":
    try:
        import importlib
        chat_module = importlib.import_module("chat")
        importlib.reload(chat_module)
        if hasattr(chat_module, 'run'):
            chat_module.run()
        else:
            st.error('Chat module does not expose run()')
    except Exception as e:
        st.error(f"Failed to load chat module: {e}")
    st.stop()


# sidebar for navigation
with st.sidebar:
    
    selected = option_menu(
    'Menu',
    [
        'Multiple Disease Prediction',
        'Diabetes',
        'Heart Disease',
        'Parkinsons',
        'Breast Cancer',
        'Pneumonia',
        'Eye Disease',
        'Kidney',
        'Stroke Prediction',
        'Lung Cancer'
        # 'Malaria'
    ],
    icons=[
        'hospital',
        'droplet',
        'heart-pulse',
        'person-bounding-box',
        'person-heart',
        'lungs',
        'eye',
        'capsule',
        'activity',
        'lungs-fill',
        # 'bug'
    ],
    default_index=0
)

    # st.markdown('<br/>')
    import streamlit.components.v1 as components
    # Link to an external chat server; start it with the command shown below
    components.html(
"""
<style>
.chat-btn {
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    border: none;
    color: white;
    padding: 14px 22px;
    font-size: 16px;
    font-weight: bold;
    border-radius: 12px;
    cursor: pointer;
    width: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 0 15px rgba(0, 114, 255, 0.6);
    position: relative;
    overflow: hidden;
}

/* Glow animation */
.chat-btn::before {
    content: "";
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(120deg, transparent, rgba(255,255,255,0.4), transparent);
    transform: rotate(25deg);
    animation: shine 3s infinite;
}

@keyframes shine {
    0% { transform: translateX(-100%) rotate(25deg); }
    100% { transform: translateX(100%) rotate(25deg); }
}

/* Hover effects */
.chat-btn:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px rgba(0, 114, 255, 0.9);
}

/* Click animation */
.chat-btn:active {
    transform: scale(0.95);
}

/* Floating pulse */
@keyframes pulse {
    0% { box-shadow: 0 0 10px rgba(0,114,255,0.5); }
    50% { box-shadow: 0 0 25px rgba(0,114,255,1); }
    100% { box-shadow: 0 0 10px rgba(0,114,255,0.5); }
}

.pulse {
    animation: pulse 2s infinite;
}

.chat-container {
    text-align: center;
}

.chat-label {
    font-size: 12px;
    margin-top: 8px;
    color: #aaa;
}
</style>

<div class="chat-container">
    <button class="chat-btn pulse" onclick="window.open('http://localhost:8502','_blank')">
        🤖 Chat with Medika
    </button>
    <div class="chat-label">
        Run: <code>streamlit run src/chat_launcher.py --server.port 8502</code>
    </div>
</div>
""",
height=120,
)
    

#MULTIPLE DISEASE PREDICTION

if (selected == 'Multiple Disease Prediction'):

    st.title("🩺 Multiple Disease Prediction System")

    patient_name = st.text_input("Patient Name")

    st.markdown("### Enter Patient Details")

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=45
        )

        smoking = st.selectbox(
            "Smoking",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        alcohol = st.selectbox(
            "Alcohol",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

    with col2:

        fatigue = st.selectbox(
            "Fatigue",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        coughing = st.selectbox(
            "Coughing",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        chest_pain = st.selectbox(
            "Chest Pain",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

    with col3:

        breath = st.selectbox(
            "Shortness of Breath",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        glucose = st.number_input(
            "Glucose Level",
            value=100
        )

        bmi = st.number_input(
            "BMI",
            value=25.0
        )

    st.divider()

    # =====================================
    # ANALYZE BUTTON
    # =====================================

    if st.button("Analyze Patient"):

        results = {}

        # =====================================
        # 🫁 LUNG CANCER PREDICTION
        # =====================================

        try:

            lung_input = np.array([[ 
                age,
                smoking,
                0,
                fatigue,
                0,
                alcohol,
                coughing,
                breath,
                chest_pain
            ]])

            lung_scaled = lung_scaler.transform(
                lung_input
            )

            lung_prob = lung_model.predict_proba(
                lung_scaled
            )[0][1]

            results["Lung Cancer"] = float(lung_prob)

        except Exception as e:

            st.error(f"Lung Model Error: {e}")

            results["Lung Cancer"] = 0.0

        # =====================================
        # 🧠 STROKE PREDICTION
        # =====================================

        try:

            stroke_input = np.array([[ 
                1,
                age,
                0,
                0,
                1,
                0,
                1,
                glucose,
                bmi,
                smoking
            ]])

            stroke_scaled = stroke_scaler.transform(
                stroke_input
            )

            stroke_prob = stroke_model.predict_proba(
                stroke_scaled
            )[0][1]

            results["Stroke"] = float(stroke_prob)

        except Exception as e:

            st.error(f"Stroke Model Error: {e}")

            results["Stroke"] = 0.0

        # =====================================
        # ❤️ HEART DISEASE PREDICTION
        # =====================================

        try:

            # Approximate medical feature generation

            sex = 1

            cp = 3 if chest_pain == 1 else 1

            trestbps = int(110 + (bmi * 1.5))

            chol = int(150 + (bmi * 3))

            fbs = 1 if glucose > 120 else 0

            restecg = 1

            thalach = int(210 - age)

            exang = chest_pain

            oldpeak = 2.0 if smoking == 1 else 0.5

            slope = 1

            ca = 0

            thal = 2

            heart_input = np.array([[
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal
            ]])

            # Preprocessing
            heart_input = heart_imputer.transform(
                heart_input
            )

            heart_input = heart_scaler.transform(
                heart_input
            )

            # Prediction
            heart_prob = heart_disease_model.predict_proba(
                heart_input
            )[0][1]

            results["Heart Disease"] = float(heart_prob)

        except Exception as e:

            st.error(f"Heart Model Error: {e}")

            results["Heart Disease"] = 0.0

        # =====================================
        # AI RECOMMENDATIONS
        # =====================================

        recommendations = []

        # Glucose
        if glucose > 140:

            recommendations.append(
                "Elevated blood sugar detected. Diabetes screening advised."
            )

        # BMI
        if bmi > 30:

            recommendations.append(
                "Obesity risk detected. Weight reduction recommended."
            )

        # Smoking
        if smoking == 1:

            recommendations.append(
                "Smoking significantly increases heart and lung disease risk."
            )

        # Alcohol
        if alcohol == 1:

            recommendations.append(
                "Reduce alcohol intake for better cardiovascular health."
            )

        # Chest pain
        if chest_pain == 1:

            recommendations.append(
                "Chest pain symptoms detected. ECG and cardiology consultation advised."
            )

        # Breathing issue
        if breath == 1:

            recommendations.append(
                "Respiratory symptoms detected. Pulmonary evaluation recommended."
            )

        # Fatigue
        if fatigue == 1:

            recommendations.append(
                "Persistent fatigue detected. Regular health monitoring advised."
            )

        # Cholesterol risk
        if bmi > 28 and smoking == 1:

            recommendations.append(
                "Possible elevated cholesterol risk detected. Lipid profile test advised."
            )

        # =====================================
        # SAVE SESSION
        # =====================================

        st.session_state["results"] = results

        st.session_state["recommendations"] = recommendations

        # =====================================
        # RISK SUMMARY
        # =====================================

        st.markdown("## 📊 Risk Summary")

        cols = st.columns(3)

        for i, (disease, prob) in enumerate(results.items()):

            with cols[i % 3]:

                if prob > 0.7:

                    st.error(
                        f"{disease}\n\n"
                        f"Risk: {prob*100:.2f}%"
                    )

                elif prob > 0.4:

                    st.warning(
                        f"{disease}\n\n"
                        f"Risk: {prob*100:.2f}%"
                    )

                else:

                    st.success(
                        f"{disease}\n\n"
                        f"Risk: {prob*100:.2f}%"
                    )

        # =====================================
        # VISUALIZATION
        # =====================================

        st.markdown("## 📈 Disease Risk Visualization")

        st.bar_chart(results)

        # =====================================
        # CLINICAL INSIGHT
        # =====================================

        max_disease = max(results, key=results.get)

        max_value = results[max_disease]

        st.markdown("## 🧠 Clinical Insight")

        if max_value > 0.7:

            st.error(
                f"⚠️ High Risk Detected: {max_disease}"
            )

        elif max_value > 0.4:

            st.warning(
                f"⚠️ Moderate Risk Detected: {max_disease}"
            )

        else:

            st.success(
                "✅ Patient appears stable"
            )

        # =====================================
        # SHOW RECOMMENDATIONS
        # =====================================

        st.markdown("## 🩺 AI Medical Recommendations")

        if len(recommendations) > 0:

            for rec in recommendations:

                st.warning(rec)

        else:

            st.success(
                "No major medical risk recommendations."
            )

        # =====================================
        # HEALTHY LIFESTYLE
        # =====================================

        st.markdown("## ❤️ Healthy Lifestyle Suggestions")

        st.info("""
        ✅ Recommended:
        - Fruits and vegetables
        - Daily walking
        - Proper hydration
        - Good sleep
        - Stress reduction

        ❌ Avoid:
        - Smoking
        - Alcohol
        - Fried foods
        - Sugary drinks
        - Junk foods
        """)

        # =====================================
        # RESULTS TABLE
        # =====================================

        st.markdown("## 📋 Prediction Table")

        st.dataframe(pd.DataFrame({
            "Disease": list(results.keys()),
            "Risk Score": [
                f"{v*100:.2f}%"
                for v in results.values()
            ]
        }))

    # =====================================
    # PDF DOWNLOAD
    # =====================================

    if "results" in st.session_state and patient_name:

        if st.button("Download PDF Report"):

            pdf_path = generate_pdf(
                patient_name,
                st.session_state["results"],
                st.session_state["recommendations"]
            )

            with open(pdf_path, "rb") as f:

                st.download_button(
                    label="Download Report",
                    data=f,
                    file_name=f"{patient_name}_report.pdf",
                    mime="application/pdf"
                )
    
if (selected == 'Diabetes'):

    st.title('Diabetes Prediction using ML')

    patient_name = st.text_input("Patient Name")

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose')
    with col3:
        BloodPressure = st.text_input('Blood Pressure')

    with col1:
        SkinThickness = st.text_input('Skin Thickness')
    with col2:
        Insulin = st.text_input('Insulin')
    with col3:
        BMI = st.text_input('BMI')

    with col1:
        DiabetesPedigreeFunction = st.text_input('DPF')
    with col2:
        Age = st.text_input('Age')

    # Store inputs
    input_dict = {
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DPF": DiabetesPedigreeFunction,
        "Age": Age
    }

    # Predict
    if st.button('Diabetes Test Result'):

        try:
            prediction = diabetes_model.predict([[
                float(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), float(Age)
            ]])

            if prediction[0] == 1:
                st.session_state["diabetes_result"] = "The person is diabetic"
            else:
                st.session_state["diabetes_result"] = "The person is not diabetic"

            st.session_state["diabetes_inputs"] = input_dict

        except:
            st.session_state["diabetes_result"] = "Invalid input"

    # Show result
    if "diabetes_result" in st.session_state:
        st.success(st.session_state["diabetes_result"])

    # Generate PDF
    if "diabetes_result" in st.session_state and patient_name:

        if st.button("Generate Diabetes Report"):

            pdf_path = generate_single_report(
                patient_name,
                "Diabetes",
                st.session_state["diabetes_inputs"],
                st.session_state["diabetes_result"]
            )

            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="Download Report",
                    data=f,
                    file_name=f"{patient_name}_diabetes_report.pdf",
                    mime="application/pdf"
                )




# ==========================================
# ❤️ HEART DISEASE PREDICTION (XGBOOST)
# ==========================================

if (selected == 'Heart Disease'):

    st.title("❤️ Heart Disease Prediction using XGBoost")

    patient_name = st.text_input("Patient Name")

    st.markdown("### Enter Patient Details")

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            1,
            120,
            45
        )

        sex = st.selectbox(
            "Sex",
            [0, 1],
            format_func=lambda x:
            "Female" if x == 0 else "Male"
        )

        cp = st.selectbox(
            "Chest Pain Type",
            [0,1,2,3],
            help="""
                0 = Typical Angina
                1 = Atypical Angina
                2 = Non-anginal Pain
                3 = Asymptomatic
                """
        )

    with col2:

        trestbps = st.number_input(
            "Resting Blood Pressure",
            50,
            250,
            120
        )

        chol = st.number_input(
            "Cholesterol (mg/dl)",
            100,
            700,
            200
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120",
            [0,1],
            format_func=lambda x:
            "No" if x == 0 else "Yes"
        )

    with col3:

        restecg = st.selectbox(
            "Rest ECG",
            [0,1,2],
             help="""
            Resting Electrocardiogram result.

            0 = Normal
            Normal electrical activity of heart.

            1 = ST-T Wave Abnormality
            Indicates abnormal heart electrical pattern.

            2 = Left Ventricular Hypertrophy
            Enlargement/thickening of heart muscle.
            """
        )

        thalach = st.number_input(
            "Maximum Heart Rate",
            50,
            250,
            150
        )

        exang = st.selectbox(
            "Exercise Induced Angina",
            [0,1],
            format_func=lambda x:
            "No" if x == 0 else "Yes"
        )

    col4, col5, col6 = st.columns(3)

    with col4:

        oldpeak = st.number_input(
            "ST Depression",
            0.0,
            10.0,
            1.0
        )

    with col5:

        slope = st.selectbox(
            "Slope",
            [0,1,2],
            help="""
                0 = Upsloping
                Usually considered normal ECG response.
                1 = Flat
                May indicate reduced blood flow to heart.
                2 = Downsloping
                Often associated with higher heart disease risk.
                """
        )

    with col6:

        ca = st.selectbox(
            "Major Vessels Colored",
            [0,1,2,3],
            help="""
            Number of major blood vessels detected by fluoroscopy.

            0 = No major blockage
            1 = One vessel affected
            2 = Two vessels affected
            3 = Three vessels affected

            Higher values indicate higher cardiac risk.
            """
        )

    thal = st.selectbox(
    "Thal",
    [0,1,2,3],
    help="""
    Thallium stress test result.

    0 = Normal Blood Flow
    1 = Fixed Defect
    2 = Reversible Defect
    3 = Severe Abnormality

    Used to detect blood flow problems in heart muscles.
    """
    )

    # =====================================
    # HEART TEST BUTTON
    # =====================================

    if st.button("Heart Test Result"):

        try:

            # =====================================
            # INPUT ARRAY
            # =====================================

            input_data = np.array([[
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal
            ]])

            # =====================================
            # PREPROCESSING
            # =====================================

            input_data = heart_imputer.transform(
                input_data
            )

            input_data = heart_scaler.transform(
                input_data
            )

            # =====================================
            # PREDICTION
            # =====================================

            prediction = heart_disease_model.predict(
                input_data
            )[0]

            probability = heart_disease_model.predict_proba(
                input_data
            )[0][1]

            # =====================================
            # RESULT DISPLAY
            # =====================================

            st.markdown("## Prediction Result")

            if prediction == 1:

                st.error(
                    f"⚠️ Heart Disease Detected\n\n"
                    f"Risk Probability: {probability*100:.2f}%"
                )

            else:

                st.success(
                    f"✅ No Heart Disease Detected\n\n"
                    f"Confidence: {(1-probability)*100:.2f}%"
                )

            # =====================================
            # RISK METER
            # =====================================

            st.markdown("### Risk Meter")

            st.progress(float(probability))

            # =====================================
            # CLINICAL ANALYSIS
            # =====================================

            st.markdown("## Clinical Analysis")

            if probability > 0.75:

                st.error("""
                🔴 HIGH CARDIAC RISK

                Immediate cardiology consultation recommended.
                ECG / Echo test strongly advised.
                Patient may require urgent monitoring.
                """)

            elif probability > 0.45:

                st.warning("""
                🟠 MODERATE CARDIAC RISK

                Lifestyle modification recommended.
                Regular BP and cholesterol monitoring advised.
                Exercise and diet control needed.
                """)

            else:

                st.success("""
                🟢 LOW CARDIAC RISK

                Maintain healthy lifestyle.
                Continue regular exercise and balanced diet.
                """)

            # =====================================
            # AI RECOMMENDATIONS
            # =====================================

            st.markdown("## 🩺 AI Medical Recommendations")

            recommendations = []

            # Blood Pressure
            if trestbps > 140:

                recommendations.append(
                    "High blood pressure detected. Reduce salt intake and monitor BP regularly."
                )

            # Cholesterol
            if chol > 240:

                recommendations.append(
                    "High cholesterol detected. Avoid oily foods and maintain healthy diet."
                )

            # Heart Rate
            if thalach < 100:

                recommendations.append(
                    "Low heart performance detected. Cardiology consultation advised."
                )

            # Exercise Angina
            if exang == 1:

                recommendations.append(
                    "Exercise-induced angina symptoms detected. Avoid heavy exertion."
                )

            # Diabetes Risk
            if fbs == 1:

                recommendations.append(
                    "Elevated blood sugar detected. Diabetes screening recommended."
                )

            if len(recommendations) > 0:

                for rec in recommendations:

                    st.warning(rec)

            else:

                st.success(
                    "No major medical risks detected."
                )

            # =====================================
            # HEART DIET
            # =====================================

            st.markdown("## ❤️ Recommended Heart Diet")

            st.info("""
            Recommended:
            - Fruits and vegetables
            - Oats and whole grains
            - Fish and lean protein
            - Nuts and seeds
            - Daily walking

            Avoid:
            - Fried foods
            - Junk foods
            - Excess salt
            - Sugary drinks
            - Smoking
            """)

            # =====================================
            # INPUT DICTIONARY
            # =====================================

            input_dict = {

                "Age": age,

                "Sex":
                "Male" if sex == 1 else "Female",

                "Chest Pain Type": cp,

                "Resting Blood Pressure":
                trestbps,

                "Cholesterol":
                chol,

                "Fasting Blood Sugar":
                fbs,

                "Rest ECG":
                restecg,

                "Maximum Heart Rate":
                thalach,

                "Exercise Angina":
                exang,

                "Oldpeak":
                oldpeak,

                "Slope":
                slope,

                "Major Vessels":
                ca,

                "Thal":
                thal
            }

            # =====================================
            # SAVE SESSION
            # =====================================

            st.session_state["heart_result"] = prediction

            st.session_state["heart_probability"] = probability

            st.session_state["heart_inputs"] = input_dict

            st.session_state["heart_recommendations"] = recommendations

        except Exception as e:

            st.error(
                f"Prediction Error: {str(e)}"
            )

    # =====================================
    # HEART REPORT GENERATION
    # =====================================

    if (
        "heart_result" in st.session_state
        and patient_name
    ):

        if st.button("Generate Heart Report"):

            pdf_path = generate_heart_report(
                patient_name,
                st.session_state["heart_inputs"],
                st.session_state["heart_result"],
                st.session_state["heart_probability"],
                st.session_state["heart_recommendations"]
            )

            with open(pdf_path, "rb") as f:

                st.download_button(
                    label="Download Heart Report",
                    data=f,
                    file_name=f"{patient_name}_heart_report.pdf",
                    mime="application/pdf"
                )
    

# Parkinson's Prediction Page
if (selected == "Parkinsons"):

    st.title("Parkinson's Disease Prediction")

    patient_name = st.text_input("Patient Name")

    col1, col2, col3, col4, col5 = st.columns(5)

    fo = st.text_input('Fo')
    fhi = st.text_input('Fhi')
    flo = st.text_input('Flo')
    Jitter_percent = st.text_input('Jitter%')
    Jitter_Abs = st.text_input('Jitter Abs')

    RAP = st.text_input('RAP')
    PPQ = st.text_input('PPQ')
    DDP = st.text_input('DDP')
    Shimmer = st.text_input('Shimmer')
    Shimmer_dB = st.text_input('Shimmer dB')

    APQ3 = st.text_input('APQ3')
    APQ5 = st.text_input('APQ5')
    APQ = st.text_input('APQ')
    DDA = st.text_input('DDA')
    NHR = st.text_input('NHR')

    HNR = st.text_input('HNR')
    RPDE = st.text_input('RPDE')
    DFA = st.text_input('DFA')
    spread1 = st.text_input('spread1')
    spread2 = st.text_input('spread2')

    D2 = st.text_input('D2')
    PPE = st.text_input('PPE')

    input_dict = {
        "fo": fo, "fhi": fhi, "flo": flo,
        "Jitter%": Jitter_percent, "JitterAbs": Jitter_Abs,
        "RAP": RAP, "PPQ": PPQ, "DDP": DDP,
        "Shimmer": Shimmer, "Shimmer_dB": Shimmer_dB,
        "APQ3": APQ3, "APQ5": APQ5, "APQ": APQ,
        "DDA": DDA, "NHR": NHR, "HNR": HNR,
        "RPDE": RPDE, "DFA": DFA,
        "spread1": spread1, "spread2": spread2,
        "D2": D2, "PPE": PPE
    }

    if st.button("Parkinson Test Result"):

        try:
            values = list(map(float, input_dict.values()))
            pred = parkinsons_model.predict([values])[0]

            if pred == 1:
                st.session_state["parkinson_result"] = "Parkinson's Disease Detected"
            else:
                st.session_state["parkinson_result"] = "No Parkinson's Disease"

            st.session_state["parkinson_inputs"] = input_dict

        except:
            st.session_state["parkinson_result"] = "Invalid Input"

    if "parkinson_result" in st.session_state:
        st.success(st.session_state["parkinson_result"])

    if "parkinson_result" in st.session_state and patient_name:
        if st.button("Generate Parkinson Report"):

            pdf_path = generate_single_report(
                patient_name,
                "Parkinson's Disease",
                st.session_state["parkinson_inputs"],
                st.session_state["parkinson_result"]
            )

            with open(pdf_path, "rb") as f:
                st.download_button("Download Report", f, file_name="parkinson_report.pdf")


# Breast Cancer Prediction Page (FINAL HYBRID)
if (selected == 'Breast Cancer'):

    st.title("Breast Cancer Prediction 🧬")

    import tensorflow as tf

    # 🔥 Choose mode
    mode = st.radio(
        "Choose Prediction Type:",
        ["Clinical Data (ML)", "Image (CNN)"]
    )

    # ============================
    # 🔹 OPTION 1: CLINICAL DATA
    # ============================
    if mode == "Clinical Data (ML)":

        st.subheader("Enter Clinical Features")

        input_data = []
        cols = st.columns(3)

        for i, feature in enumerate(breast_features):
            with cols[i % 3]:
                val = st.number_input(feature, value=0.0)
                input_data.append(val)

        result = ""

        if st.button("Predict (Clinical Data)"):

            try:
                input_array = np.array(input_data).reshape(1, -1)

                # Scale
                input_scaled = breast_scaler.transform(input_array)

                # Predict
                prediction = breast_model.predict(input_scaled)[0]
                prob = breast_model.predict_proba(input_scaled)[0][1]

                if prediction == 1:
                    result = f"⚠️ Malignant (Cancer) - Risk: {prob:.2f}"
                else:
                    result = f"✅ Benign - Confidence: {1 - prob:.2f}"

            except Exception as e:
                result = f"Error: {str(e)}"

        st.success(result)

    # ============================
    # 🔹 OPTION 2: IMAGE MODEL
    # ============================
    else:

        st.subheader("Upload Image")

        @st.cache_resource
        def load_breast_image_model():

            # 🔥 Fix Keras compatibility issue
            class DenseCompat(tf.keras.layers.Dense):
                @classmethod
                def from_config(cls, config):
                    config = dict(config)
                    config.pop("quantization_config", None)
                    return super().from_config(config)

            # ✅ Since FFD.keras is in SAME folder as app.py
            model_path = os.path.join(os.path.dirname(__file__), "FFD.keras")

            return tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects={"Dense": DenseCompat}
            )

        model = load_breast_image_model()

        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:

            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Preprocess
            img = image.resize((150, 150))
            img_array = np.array(img).astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            pred = model.predict(img_array)[0][0]

            st.write("Raw Prediction:", float(pred))  # debug

            if pred > 0.5:
                st.error(f"⚠️ Malignant (Confidence: {pred:.2f})")
            else:
                st.success(f"✅ Benign (Confidence: {1 - pred:.2f})")

# Kidney Disease Prediction Page
if (selected == 'Kidney'):

    st.title('Kidney Disease Prediction')

    patient_name = st.text_input("Patient Name")

    kidney_model, kidney_features, kidney_scaler, kidney_encoders = load_kidney_artifacts()

    input_data = {}

    for feature in kidney_features:
        if feature in kidney_encoders:
            val = st.selectbox(feature, kidney_encoders[feature].classes_)
        else:
            val = st.number_input(feature)

        input_data[feature] = val

    if st.button('Kidney Test Result'):

        try:
            for col, le in kidney_encoders.items():
                input_data[col] = le.transform([input_data[col]])[0]

            arr = np.array([list(input_data.values())])
            arr = kidney_scaler.transform(arr)

            pred = kidney_model.predict(arr)[0]

            if pred == 1:
                st.session_state["kidney_result"] = "Kidney Disease Detected"
            else:
                st.session_state["kidney_result"] = "No Kidney Disease"

            st.session_state["kidney_inputs"] = input_data

        except:
            st.session_state["kidney_result"] = "Error"

    if "kidney_result" in st.session_state:
        st.success(st.session_state["kidney_result"])

    if "kidney_result" in st.session_state and patient_name:
        if st.button("Generate Kidney Report"):

            pdf_path = generate_single_report(
                patient_name,
                "Kidney Disease",
                st.session_state["kidney_inputs"],
                st.session_state["kidney_result"]
            )

            with open(pdf_path, "rb") as f:
                st.download_button("Download Report", f, file_name="kidney_report.pdf")

if (selected == 'Eye Disease'):

    st.title("Eye Disease Prediction 👁️")

    eye_model = load_eye_model()

    IMG_SIZE = (256, 256)  # ✅ from your model

    uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess: match training preprocessing as closely as possible
        img_resized = image.resize(IMG_SIZE)
        img_array = np.array(img_resized).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        import tensorflow as tf
        raw_pred = eye_model.predict(img_array)

        class_names = [
            'Normal',
            'Cataract',
            'Diabetic Retinopathy',
            'Glaucoma'
        ]

        # Debug info: show raw prediction, shape and dtype
        st.write('raw_prediction:', raw_pred)
        st.write('shape:', raw_pred.shape, 'dtype:', raw_pred.dtype)

        # If outputs are logits (not summing to ~1), convert with softmax
        if raw_pred.ndim == 2 and not np.allclose(raw_pred.sum(axis=1), 1.0, atol=1e-3):
            probs = tf.nn.softmax(raw_pred, axis=1).numpy()[0]
        else:
            probs = raw_pred[0]

        # Show per-class probabilities for debugging
        for name, p in zip(class_names, probs):
            st.write(f"{name}: {p:.4f}")

        idx = int(np.argmax(probs))
        predicted_class = class_names[idx]
        confidence = float(probs[idx])

        # Output
        if predicted_class == "Normal":
            st.success(f"✅ {predicted_class} (confidence: {confidence:.2f})")
        else:
            st.error(f"⚠️ {predicted_class} (confidence: {confidence:.2f})")

        # Additional quick test: try channel-swapped input (BGR) to see effect
        img_bgr = img_array[..., ::-1]
        pred_bgr = eye_model.predict(img_bgr)
        if pred_bgr.ndim == 2 and not np.allclose(pred_bgr.sum(axis=1), 1.0, atol=1e-3):
            probs_bgr = tf.nn.softmax(pred_bgr, axis=1).numpy()[0]
        else:
            probs_bgr = pred_bgr[0]
        st.write('predictions (BGR swap):', probs_bgr)

        patient_name = st.text_input("Patient Name")

        st.session_state["eye_result"] = predicted_class

        if "eye_result" in st.session_state and patient_name:
            if st.button("Generate Eye Report"):
                pdf_path = generate_single_report(
                            patient_name,
                    "Eye Disease",
                    {"Image": "Uploaded Eye Image"},
                st.session_state["eye_result"]
                )

# Stroke Prediction Page
if (selected == 'Stroke Prediction'):

    st.title('Stroke Prediction')

    patient_name = st.text_input("Patient Name")

    gender = st.selectbox('Gender', ['Male','Female','Other'])
    age = st.number_input('Age')
    hypertension = st.selectbox('Hypertension',[0,1])
    heart_disease = st.selectbox('Heart Disease',[0,1])

    ever_married = st.selectbox('Married',['Yes','No'])
    work_type = st.selectbox('Work Type',['Private','Self-employed','Govt_job','children','Never_worked'])
    residence = st.selectbox('Residence',['Urban','Rural'])

    glucose = st.number_input('Glucose')
    bmi = st.number_input('BMI')
    smoking = st.selectbox('Smoking',['formerly smoked','never smoked','smokes','Unknown'])

    input_dict = {
        "gender": gender, "age": age,
        "hypertension": hypertension, "heart_disease": heart_disease,
        "married": ever_married, "work": work_type,
        "residence": residence, "glucose": glucose,
        "bmi": bmi, "smoking": smoking
    }

    if st.button('Stroke Test Result'):

        try:
            # encode
            g = stroke_encoders["gender"].transform([gender])[0]
            m = stroke_encoders["ever_married"].transform([ever_married])[0]
            w = stroke_encoders["work_type"].transform([work_type])[0]
            r = stroke_encoders["Residence_type"].transform([residence])[0]
            s = stroke_encoders["smoking_status"].transform([smoking])[0]

            arr = np.array([[g, age, hypertension, heart_disease, m, w, r, glucose, bmi, s]])
            arr = stroke_scaler.transform(arr)

            pred = stroke_model.predict(arr)[0]

            if pred == 1:
                st.session_state["stroke_result"] = "High Risk Stroke"
            else:
                st.session_state["stroke_result"] = "Low Risk Stroke"

            st.session_state["stroke_inputs"] = input_dict

        except:
            st.session_state["stroke_result"] = "Error"

    if "stroke_result" in st.session_state:
        st.success(st.session_state["stroke_result"])

    if "stroke_result" in st.session_state and patient_name:
        if st.button("Generate Stroke Report"):

            pdf_path = generate_single_report(
                patient_name,
                "Stroke",
                st.session_state["stroke_inputs"],
                st.session_state["stroke_result"]
            )

            with open(pdf_path, "rb") as f:
                st.download_button("Download Report", f, file_name="stroke_report.pdf")

# Pneumonia Prediction Page
if (selected == 'Pneumonia'):

    st.title("Pneumonia Detection 🫁")

    import tensorflow as tf

    @st.cache_resource
    def load_pneumonia_model():

        # ✅ Fix for Dense layer error (quantization_config issue)
        class DenseCompat(tf.keras.layers.Dense):
            @classmethod
            def from_config(cls, config):
                config = dict(config)
                config.pop("quantization_config", None)
                return super().from_config(config)

        # ✅ Fix for Resizing layer issue (safe fallback)
        class ResizingCompat(tf.keras.layers.Resizing):
            @classmethod
            def from_config(cls, config):
                config = dict(config)
                config.pop("antialias", None)
                return super().from_config(config)

        custom_objects = {
            "Dense": DenseCompat,
            "Resizing": ResizingCompat
        }

        # ✅ Correct path (relative to project root)
        model_path = os.path.join(BASE_DIR, "models", "pnue.h5")

        return tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=custom_objects
        )

    model = load_pneumonia_model()

    IMG_SIZE = (150, 150)

    uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # ✅ Preprocessing (same as training)
        img_resized = image.resize(IMG_SIZE)
        img_array = np.array(img_resized).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ✅ Prediction
        prediction = model.predict(img_array)[0][0]

        st.write("Raw Prediction:", float(prediction))  # debug

        if prediction > 0.5:
            result = "PNEUMONIA"
            confidence = prediction
            st.error(f"⚠️ {result} (confidence: {confidence:.2f})")
        else:
            result = "NORMAL"
            confidence = 1 - prediction
            st.success(f"✅ {result} (confidence: {confidence:.2f})")
        patient_name = st.text_input("Patient Name")

# After prediction
        st.session_state["pneumonia_result"] = result

        if "pneumonia_result" in st.session_state and patient_name:
            if st.button("Generate Pneumonia Report"):
                pdf_path = generate_single_report(
                patient_name,
                "Pneumonia",
                {"Image": "Uploaded X-ray"},
                st.session_state["pneumonia_result"]
            )

# Lung Cancer Prediction Page (FINAL)
if (selected == 'Lung Cancer'):

    st.title("Lung Cancer Prediction")

    patient_name = st.text_input("Patient Name")

    input_data = []

    cols = st.columns(3)

    for i, feature in enumerate(lung_features):
        with cols[i % 3]:
            val = st.number_input(feature, value=0)
            input_data.append(val)

    input_dict = dict(zip(lung_features, input_data))

    if st.button("Predict"):

        arr = np.array(input_data).reshape(1,-1)
        arr = lung_scaler.transform(arr)

        pred = lung_model.predict(arr)[0]
        prob = lung_model.predict_proba(arr)[0][1]

        if pred == 1:
            st.session_state["lung_result"] = f"High Risk ({prob:.2f})"
        else:
            st.session_state["lung_result"] = f"Low Risk ({1-prob:.2f})"

        st.session_state["lung_inputs"] = input_dict

    if "lung_result" in st.session_state:
        st.success(st.session_state["lung_result"])

    if "lung_result" in st.session_state and patient_name:
        if st.button("Generate Lung Report"):

            pdf_path = generate_single_report(
                patient_name,
                "Lung Cancer",
                st.session_state["lung_inputs"],
                st.session_state["lung_result"]
            )

            with open(pdf_path, "rb") as f:
                st.download_button("Download Report", f, file_name="lung_report.pdf")