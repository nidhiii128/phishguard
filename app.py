import streamlit as st
import joblib
import pandas as pd
import numpy as np
from src.feature_engineering import URLFeatureExtractor
import plotly.graph_objects as go
import plotly.express as px

# --- Configuration and Styling ---

# Set page configuration - Removed initial_sidebar_state
st.set_page_config(
    page_title="PhishGuard",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS for a professional, consolidated GUI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&display=swap');

   
    .stApp {
        background-image: url("https://raw.githubusercontent.com/nidhiii128/phishing-image-/main/download.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        position: relative;
        z-index: 0;
        font-family: 'Playfair Display', serif;
        color: #f5f5f5; /* Light text on dark overlay */
    }
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.6); /* Adjust transparency (0.6 = 60%) */
        z-index: 0;
    }

    /* Ensure content is above the overlay */
    .stApp > * {
        position: relative;
        z-index: 1;

    }


h1, h2, h3, h4, h5, h6, p, span, div {
        font-family: 'Playfair Display', serif !important;
        background: linear-gradient(90deg, #ffffff, #e0e0e0);
        -webkit-background-clip: text;
        color: white; /* fallback */
    }


    .main-header {
        font-size: 1rem;
        color: #00ffff;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0px 0px 10px rgba(0,255,255,0.6);

    }


    /* Optional hover glow for interactivity */
    .main-header:hover {
        text-shadow: 0px 0px 20px rgba(0,255,255,0.9);
        transition: all 0.3s ease;
    }

     div.stButton > button {
        background-color: #006A67 !important;
        color: #000000 !important;
        font-family: 'Playfair Display', serif;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 5px 15px rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
        cursor: pointer;
    }

    div.stButton > button:hover {
        background-color: #26667F !important;
        color: #111111 !important;
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 20px rgba(255, 255, 255, 0.6);
    }

    /* Optional: Glassmorphism for main container */
    .block-container {
        padding: 2rem;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        box-shadow: 0 0 20px rgba(0,0,0,0.4);
    }

    /* Main container styling for the 'one div' look */
    
    .sub-header {
        font-size: 1.25rem;
        color: #5a6270;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Input and Button Styling */
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 2px solid #ced4da;
    }


    .phishing-box {
        background-color: #ffe6e6; /* Light red */
        border: 1px solid #e9b3b3;
        color: #8c1a1a;
    }
    
    /* Feature Metrics */
    [data-testid="stMetric"] {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    /* Remove the default Streamlit sidebar styling (not needed, but good practice if it reappeared) */
    [data-testid="stSidebar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# --- Utility Functions (unchanged) ---

@st.cache_resource
def load_model():
    """Load the trained model and feature extractor"""
    try:
        model_data = joblib.load('models/trained_model.joblib')
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {e}. Using placeholder data for UI demonstration.")
        return {
            'model': None, 'model_name': 'RandomForestClassifier', 'needs_scaling': False,
            'feature_columns': [],
            'performance': {'accuracy': 0.952, 'precision': 0.945, 'recall': 0.960, 'f1': 0.952}
        }

def predict_url(url, model_data):
    """Predict if a URL is phishing or benign"""
    extractor = URLFeatureExtractor()
    
    if model_data.get('model') is None:
        return 0, np.array([0.9, 0.1]), extractor.extract_features(url)

    features = extractor.extract_features(url)
    feature_df = pd.DataFrame([features])
    
    for col in model_data['feature_columns']:
        if col not in feature_df.columns:
            feature_df[col] = 0
    feature_df = feature_df[model_data['feature_columns']]
    
    if model_data['needs_scaling']:
        features_scaled = model_data['scaler'].transform(feature_df)
        prediction = model_data['model'].predict(features_scaled)[0]
        probability = model_data['model'].predict_proba(features_scaled)[0]
    else:
        prediction = model_data['model'].predict(feature_df)[0]
        probability = model_data['model'].predict_proba(feature_df)[0]
    
    return prediction, probability, features

# --- Main Application Logic ---

def main():
    # Load model
    model_data = load_model()
    if model_data is None:
        st.error("Model could not be loaded. Please check if the model file exists.")
        return

    # Use a single container to wrap all main content
    with st.container():
        st.markdown('<div class="main-content-container">', unsafe_allow_html=True)
        
        # --- 1. Header Section ---
        st.markdown('<h1 class="main-header">PhishGuard 🛡️</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">A real-time phishing URL scanner powered by Machine Learning</p>', unsafe_allow_html=True)

        # --- 2. Scanner Section (Top) ---
        st.header("URL Scanner")
        
        url_input = st.text_input(
            "Enter a URL to analyze:",
            placeholder="https://www.example.com",
            key="url_input_field",
            help="Paste the complete URL including http:// or https://"
        )
        
        # Handle test URL injection
        if 'test_url' in st.session_state:
            url_input = st.session_state.test_url
            del st.session_state.test_url
        
        analyze_btn = st.button("Scan URL", type="primary", use_container_width=True)
        
        # --- 3. Analysis Results Section (Below scanner) ---
        
        if analyze_btn and url_input:
            if not url_input.startswith(('http://', 'https://')):
                st.warning("Assuming 'https://' prefix for analysis.")
                url_to_analyze = 'https://' + url_input
            else:
                url_to_analyze = url_input
            
            # Prediction Logic
            with st.spinner("Analyzing URL features..."):
                prediction, probability, features = predict_url(url_to_analyze, model_data)
            
            st.subheader("Analysis Results")
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                # Confidence gauge
                confidence = probability[1] if prediction == 1 else probability[0]
                color_scheme = ['#e63946', '#f4a261', '#26667F'] 
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = confidence * 100,
                    title = {'text': "Confidence Score"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#124170"},
                        'steps': [
                            {'range': [0, 50], 'color': color_scheme[0]},
                            {'range': [50, 80], 'color': color_scheme[1]},
                            {'range': [80, 100], 'color': color_scheme[2]}
                        ],
                    }
                ))
                fig.update_layout(height=280, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)
            
            with col_res2:
                if prediction == 0:  # Benign
                    st.markdown('<div class="safe-box">', unsafe_allow_html=True)
                    st.write("**VERDICT: SAFE URL**")
                    st.write("This URL appears to be **BENIGN** and structurally safe.")
                    st.write(f"**Confidence:** `{probability[0]*100:.1f}%`")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:  # Phishing
                    
                    st.write("**VERDICT: PHISHING URL**")
                    st.write("This URL is **SUSPICIOUS**! Exercise extreme caution.")
                    st.write(f"**Confidence:** `{probability[1]*100:.1f}%`")
                    st.markdown("**Action:** Do not enter any personal or financial information!")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            
            # Detailed Feature Analysis
            st.subheader(" Feature Indicators")
            
            char_count = sum(features.get(f'count_{char}', 0) for char in 
                            ['dot', 'hyphen', 'underscore', 'slash', 'questionmark', 
                             'equal', 'at', 'and', 'exclamation', 'space', 'tilde', 
                             'comma', 'plus', 'asterisk', 'hash', 'dollar', 'percent'])

            key_features = {
                'URL Length (Chars)': features['url_length'],
                'Special Chars (Count)': char_count,
                'Suspicious Words (Count)': features.get('suspicious_words_count', 0),
                'IP Address Used?': 'Yes' if features.get('ip_in_url') else 'No',
                'URL Shortener Used?': 'Yes' if features.get('is_shortened') else 'No',
                'Uses HTTPS?': 'Yes' if features.get('uses_https') else 'No'
            }
            
            feat_cols = st.columns(3)
            for i, (feature, value) in enumerate(key_features.items()):
                with feat_cols[i % 3]:
                    st.metric(feature, value)

        # --- 4. Model Overview and Test Examples (Below all results) ---
        
        
        # New location for "About PhishGuard" content
        col_about, col_model = st.columns(2)
        
        with col_about:
            st.subheader("About PhishGuard")
            st.markdown(f"""
            This tool analyzes URLs for phishing patterns using a **{model_data['model_name']}** model
            trained on 500,000+ examples.
            
            **How it works:**
            - Analyzes URL structure and patterns
            - Uses lexical features for safety
            - Provides instant results
            """)
            st.markdown("**Disclaimer:** This tool is for educational purposes. Always use multiple security measures.")

        with col_model:
            st.subheader("Model Performance")
            
            # Performance Bar Chart
            perf_data = {
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'Score': [
                    model_data['performance']['accuracy'],
                    model_data['performance']['precision'],
                    model_data['performance']['recall'],
                    model_data['performance']['f1']
                ]
            }
            perf_df = pd.DataFrame(perf_data)
            perf_df['Score'] = (perf_df['Score'] * 100).round(1)
            
            fig = px.bar(perf_df, x='Score', y='Metric', orientation='h',
                         title="Validation Performance (%)",
                         color='Score', color_continuous_scale='Mint')
            fig.update_layout(title_x=0.5, height=250, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)


     
if __name__ == "__main__":
    main()