import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="GlucoGuard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    .input-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .section-title {
        color: #2c3e50;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(255,107,107,0.3);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(81,207,102,0.3);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        text-align: center;
        border-top: 4px solid #667eea;
    }
    
    .info-card {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .predict-button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        margin: 1rem 0;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #51cf66, #ff6b6b);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa, #e9ecef);
    }
    
    .feature-explanation {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #667eea;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #495057;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler (with error handling)
@st.cache_resource
def load_models():
    try:
        model = joblib.load("final_xgb_diabetes_model.pkl")
        scaler = joblib.load("scaler_app.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please ensure 'final_xgb_diabetes_model.pkl' and 'scaler_app.pkl' are in the same directory.")
        return None, None

model, scaler = load_models()

# Features configuration
features_to_scale = ["GenHlth", "BMI", "Age", "Income"]
top_features = ['GenHlth', 'BMI', 'Age', 'HighBP', 'HighChol', 'Income', 'Sex', 'HeartDiseaseorAttack']

# Enhanced feature configurations with better descriptions
feature_configs = {
    "GenHlth": {
        "label": "General Health Status",
        "description": "How would you rate your overall health?",
        "options": {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"},
        "icon": "🏥"
    },
    "BMI": {
        "label": "Body Mass Index (BMI)",
        "description": "Your BMI calculated from height and weight",
        "min": 12.0, "max": 98.0, "step": 0.1,
        "icon": "⚖️"
    },
    "Age": {
        "label": "Age Group",
        "description": "Select your age range",
        "options": {1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 
                   6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 
                   11: "70-74", 12: "75-79", 13: "80+"},
        "icon": "👤"
    },
    "HighBP": {
        "label": "High Blood Pressure",
        "description": "Have you been diagnosed with high blood pressure?",
        "options": {0: "No", 1: "Yes"},
        "icon": "💓"
    },
    "HighChol": {
        "label": "High Cholesterol",
        "description": "Have you been diagnosed with high cholesterol?",
        "options": {0: "No", 1: "Yes"},
        "icon": "🧪"
    },
    "Income": {
        "label": "Annual Household Income",
        "description": "Select your income bracket",
        "options": {1: "< $10,000", 2: "$10,000-$15,000", 3: "$15,000-$20,000", 
                   4: "$20,000-$25,000", 5: "$25,000-$35,000", 6: "$35,000-$50,000",
                   7: "$50,000-$75,000", 8: "$75,000+"},
        "icon": "💰"
    },
    "Sex": {
        "label": "Biological Sex",
        "description": "Select your biological sex",
        "options": {0: "Female", 1: "Male"},
        "icon": "👥"
    },
    "HeartDiseaseorAttack": {
        "label": "Heart Disease History",
        "description": "Have you ever been diagnosed with heart disease or had a heart attack?",
        "options": {0: "No", 1: "Yes"},
        "icon": "❤️"
    }
}

# Hero Section
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🩺 GlucoGuard</div>
    <div class="hero-subtitle">Advanced Machine Learning Model for Early Diabetes Risk Assessment</div>
</div>
""", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.markdown("## 📊 About This Tool")
    st.markdown("""
    This AI-powered tool uses advanced machine learning to assess your diabetes risk based on key health indicators.
    
    **🎯 Accuracy**: 85%+ prediction accuracy
    **🔬 Model**: XGBoost Classifier
    **📈 Features**: 8 key health indicators
    """)
    
    st.markdown("## ℹ️ How It Works")
    st.markdown("""
    1. **Input** your health information
    2. **AI analyzes** your risk factors
    3. **Get** personalized risk assessment
    4. **Receive** health recommendations
    """)
    
    st.markdown("## ⚠️ Important Notice")
    st.warning("This tool is for educational purposes only. Always consult healthcare professionals for medical advice.")

# Main content area
if model is None or scaler is None:
    st.stop()

# Input sections
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📝 Health Assessment Form")
    
    user_input = {}
    
    # Group inputs by category
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏥 General Health Information</div>', unsafe_allow_html=True)
    
    # General Health
    config = feature_configs["GenHlth"]
    user_input["GenHlth"] = st.select_slider(
        f"{config['icon']} {config['label']}",
        options=list(config['options'].keys()),
        format_func=lambda x: config['options'][x],
        help=config['description']
    )
    
    # BMI
    config = feature_configs["BMI"]
    col_bmi1, col_bmi2 = st.columns(2)
    with col_bmi1:
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    with col_bmi2:
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    
    calculated_bmi = weight / ((height/100) ** 2)
    user_input["BMI"] = calculated_bmi
    st.info(f"📊 Your calculated BMI: **{calculated_bmi:.1f}**")
    
    # Age
    config = feature_configs["Age"]
    user_input["Age"] = st.select_slider(
        f"{config['icon']} {config['label']}",
        options=list(config['options'].keys()),
        format_func=lambda x: config['options'][x],
        help=config['description']
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Medical History
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏥 Medical History</div>', unsafe_allow_html=True)
    
    col_med1, col_med2 = st.columns(2)
    
    with col_med1:
        config = feature_configs["HighBP"]
        user_input["HighBP"] = st.radio(
            f"{config['icon']} {config['label']}",
            options=list(config['options'].keys()),
            format_func=lambda x: config['options'][x],
            help=config['description']
        )
        
        config = feature_configs["HighChol"]
        user_input["HighChol"] = st.radio(
            f"{config['icon']} {config['label']}",
            options=list(config['options'].keys()),
            format_func=lambda x: config['options'][x],
            help=config['description']
        )
    
    with col_med2:
        config = feature_configs["HeartDiseaseorAttack"]
        user_input["HeartDiseaseorAttack"] = st.radio(
            f"{config['icon']} {config['label']}",
            options=list(config['options'].keys()),
            format_func=lambda x: config['options'][x],
            help=config['description']
        )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Demographics
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👥 Demographics</div>', unsafe_allow_html=True)
    
    col_demo1, col_demo2 = st.columns(2)
    
    with col_demo1:
        config = feature_configs["Sex"]
        user_input["Sex"] = st.radio(
            f"{config['icon']} {config['label']}",
            options=list(config['options'].keys()),
            format_func=lambda x: config['options'][x],
            help=config['description']
        )
    
    with col_demo2:
        config = feature_configs["Income"]
        user_input["Income"] = st.select_slider(
            f"{config['icon']} {config['label']}",
            options=list(config['options'].keys()),
            format_func=lambda x: config['options'][x],
            help=config['description']
        )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("## 📊 Risk Assessment")
    
    # Prediction button
    if st.button("🔍 Analyze Risk", key="predict", help="Click to get your diabetes risk assessment"):
        with st.spinner("🤖 AI is analyzing your health data..."):
            # Extract and scale relevant features
            scale_input = np.array([[user_input[feat] for feat in features_to_scale]])
            scaled_values = scaler.transform(scale_input)[0]
            
            # Reconstruct input row
            input_row = []
            for feat in top_features:
                if feat in features_to_scale:
                    input_row.append(scaled_values[features_to_scale.index(feat)])
                else:
                    input_row.append(user_input[feat])
            
            input_array = np.array(input_row).reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(input_array)[0]
            prob = model.predict_proba(input_array)[0][1]
            
            # Display results
            st.markdown("### 🎯 Your Risk Assessment")
            
            if prediction == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <h3>⚠️ Higher Risk Detected</h3>
                    <p>Risk Probability: <strong>{prob:.1%}</strong></p>
                    <p>Consider consulting a healthcare professional</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h3>✅ Lower Risk Detected</h3>
                    <p>Risk Probability: <strong>{prob:.1%}</strong></p>
                    <p>Keep maintaining your healthy lifestyle!</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk probability visualization
            st.markdown("### 📈 Risk Probability")
            
            # Create a gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Diabetes Risk %"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgreen"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk factors analysis
            st.markdown("### 🔍 Risk Factors Analysis")
            
            risk_factors = []
            if user_input["HighBP"] == 1:
                risk_factors.append("High Blood Pressure")
            if user_input["HighChol"] == 1:
                risk_factors.append("High Cholesterol")
            if user_input["HeartDiseaseorAttack"] == 1:
                risk_factors.append("Heart Disease History")
            if user_input["GenHlth"] >= 4:
                risk_factors.append("Poor General Health")
            if calculated_bmi >= 30:
                risk_factors.append("High BMI (Obesity)")
            
            if risk_factors:
                st.warning("⚠️ **Identified Risk Factors:**")
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.success("✅ **No major risk factors identified!**")
            
            # Recommendations
            st.markdown("### 💡 Personalized Recommendations")
            
            recommendations = []
            if calculated_bmi >= 25:
                recommendations.append("🏃‍♂️ Consider weight management through diet and exercise")
            if user_input["GenHlth"] >= 3:
                recommendations.append("🏥 Regular health check-ups recommended")
            if user_input["HighBP"] == 1 or user_input["HighChol"] == 1:
                recommendations.append("💊 Follow prescribed medication regimen")
            
            recommendations.extend([
                "🥗 Maintain a balanced, low-sugar diet",
                "🚶‍♀️ Regular physical activity (150 min/week)",
                "😴 Ensure adequate sleep (7-9 hours)",
                "🧘‍♀️ Manage stress through relaxation techniques"
            ])
            
            for rec in recommendations:
                st.write(f"• {rec}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>🩺 AI Diabetes Risk Predictor</strong></p>
    <p>Powered by Advanced Machine Learning • Built with ❤️ for Better Health</p>
    <p><em>Last updated: {}</em></p>
    <p style='font-size: 0.8rem; margin-top: 1rem;'>
        ⚠️ <strong>Disclaimer:</strong> This tool is for educational and informational purposes only. 
        It should not replace professional medical advice, diagnosis, or treatment.
    </p>
</div>
""".format(datetime.now().strftime("%B %Y")), unsafe_allow_html=True)