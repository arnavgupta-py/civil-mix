import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Concrete Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif !important;
    }

    .stApp {
        background-color: #0d1117;
        background-image: radial-gradient(circle at 15% 50%, rgba(20, 150, 220, 0.06), transparent 35%),
                          radial-gradient(circle at 85% 30%, rgba(130, 20, 220, 0.06), transparent 35%);
        color: #c9d1d9;
    }

    .header-box {
        padding: 3.5rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 100%);
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(12px);
        margin-bottom: 2.5rem;
        text-align: center;
        animation: slideDown 0.6s ease-out;
    }

    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(to right, #4ca1af, #c4e0e5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.75rem;
        letter-spacing: -1px;
    }

    .sub-title {
        font-size: 1.25rem;
        color: #8b949e;
        font-weight: 400;
    }

    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(255, 255, 255, 0.02);
        border-radius: 8px;
        padding: 4px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 24px;
        border-radius: 6px;
        border: none !important;
        background-color: transparent;
        color: #8b949e;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #fff;
        background-color: rgba(255, 255, 255, 0.05);
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(76, 161, 175, 0.15) !important;
        color: #c4e0e5 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }

    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3.5rem;
        background: linear-gradient(135deg, #1b3a4b, #203a43, #2c5364);
        border: 1px solid #4ca1af;
        color: white;
        font-size: 1.15rem;
        font-weight: 600;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton > button:hover {
        border: 1px solid #c4e0e5;
        box-shadow: 0 0 15px rgba(76, 161, 175, 0.5);
        color: #ffffff;
        transform: translateY(-2px);
    }
    
    .result-card {
        padding: 2.5rem;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(76, 161, 175, 0.1) 0%, rgba(196, 224, 229, 0.05) 100%);
        border: 1px solid rgba(76, 161, 175, 0.4);
        text-align: center;
        margin-top: 1.5rem;
        animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    .result-value {
        font-size: 4rem;
        font-weight: 700;
        color: #4ca1af;
        margin: 1rem 0;
        text-shadow: 0 0 20px rgba(76, 161, 175, 0.3);
    }
    .result-label {
        color: #c4e0e5;
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }

    .info-box {
        background-color: rgba(56, 139, 253, 0.1);
        border-left: 4px solid #58a6ff;
        padding: 1.25rem;
        border-radius: 6px;
        color: #c9d1d9;
        margin-bottom: 2rem;
        font-size: 1.05rem;
        line-height: 1.6;
    }

    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes popIn {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = joblib.load('assests/models/lasso_model.pkl')
    scaler = joblib.load('assests/models/scaler.pkl')
    return model, scaler

model, scaler = load_assets()

st.markdown("""
<div class="header-box">
    <div class="main-title">Concrete Strength Intelligence</div>
    <div class="sub-title">Advanced predictive modeling for 28-day compressive strength based on mix design parameters</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Prediction Engine", "Comparative Analysis Report"])

with tab1:
    st.markdown("### Enter Mix Proportions")
    
    with st.container(border=True):
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            cement = st.number_input("Cement (kg/m³)", value=350.0, step=10.0, format="%.1f")
            fly_ash = st.number_input("Fly Ash (kg/m³)", value=110.0, step=5.0, format="%.1f")
            fine_agg = st.number_input("Fine Aggregate (kg/m³)", value=450.0, step=10.0, format="%.1f")
            coarse_agg = st.number_input("Coarse Aggregate (kg/m³)", value=600.0, step=10.0, format="%.1f")
            
        with col2:
            crush_sand = st.number_input("Crush Sand (kg/m³)", value=800.0, step=10.0, format="%.1f")
            wc_ratio = st.number_input("Water-Cement Ratio", value=0.35, step=0.01, format="%.2f")
            water = st.number_input("Water (kg/m³)", value=160.0, step=5.0, format="%.1f")
            admixture = st.number_input("Admixture (kg/m³)", value=5.0, step=0.5, format="%.1f")

    st.write("")

    if st.button("Calculate Predicted Strength", width="stretch"):
        input_df = pd.DataFrame(
            [[cement, fly_ash, fine_agg, coarse_agg, crush_sand, wc_ratio, water, admixture]], 
            columns=['Cement (Kg/m3)', 'Fly ash(kg/m3)', 'Fine Aggregate (kg/m3)', 
                     'Coarse Aggregate (kg/m3)', 'Crush Sand (kg/m3)', 'Water Cement Ratio', 
                     'Water (kg/m3)', 'Admixture(kg/m3)']
        )
        
        with st.spinner("Analyzing physical characteristics..."):
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)
        
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Predicted 28-Day Compressive Strength</div>
            <div class="result-value">{prediction[0]:.2f} <span style="font-size: 1.75rem; color: #8b949e;">N/mm²</span></div>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("### Model Selection Justification")
    st.markdown("""
    <div class="info-box">
        <strong>Note:</strong> Lasso Regression was selected as the optimal model. On constrained datasets (n=111), regularized linear models significantly outperform complex tree ensembles by preventing severe overfitting.
    </div>
    """, unsafe_allow_html=True)
    
    with st.container(border=True):
        col3, col4 = st.columns(2, gap="large")
        with col3:
            st.image("assests/infographics/r2_comparison.png", width="stretch", caption="R-Squared Variance Explained")
        with col4:
            st.image("assests/infographics/rmse_comparison.png", width="stretch", caption="Root Mean Squared Error")