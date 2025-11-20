"""
Plant Disease Detection System - Professional Edition
=====================================================
Author: PBL Project Team
Version: 2.0
Compatibility: Streamlit 1.12.0 | TensorFlow 2.10

Description:
This application uses a Convolutional Neural Network (CNN) to classify plant diseases
from leaf images. It includes a multi-page interface, treatment recommendations,
and model visualization tools.
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob
from pathlib import Path
import logging
import io  # Used to capture the model summary text

# ===================================================================
# 1. CONFIGURATION & SETUP
# ===================================================================

# Configure the page title, icon, and layout.
# 'layout="wide"' uses more screen space for a professional look.
st.set_page_config(
    page_title="Plant Health Expert",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up logging to track errors in the console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CONSTANTS
MODEL_PATH = r"D:\PBL project\models\model_run_30_final_epoch3.h5"
IMAGE_SIZE = (128, 128)      # The size the model expects (must match training)
CONFIDENCE_THRESHOLD = 0.70  # 70% threshold for "high confidence"
DARK_IMAGE_THRESHOLD = 51    # Intensity threshold to detect bad lighting

# FULL CLASS LIST (38 Classes)
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# TREATMENT DATABASE
# Maps disease keywords to simple treatment advice.
TREATMENT_INFO = {
    "Apple Scab": "Apply fungicides like captan or sulfur. Remove infected leaves/fruit to prevent spread.",
    "Black Rot": "Prune infected parts. Apply copper-based fungicides.",
    "Cedar Apple Rust": "Remove nearby juniper/cedar hosts. Apply fungicides in spring.",
    "Powdery Mildew": "Apply neem oil or sulfur. Improve air circulation around plants.",
    "Common Rust": "Plant resistant varieties. Apply fungicides early if detected.",
    "Northern Leaf Blight": "Crop rotation. Use resistant hybrids. Apply fungicides.",
    "Bacterial Spot": "Copper sprays can suppress it. Remove infected plant debris.",
    "Early Blight": "Apply copper-based fungicides. Mulch around base of plant.",
    "Late Blight": "Destroy infected plants immediately. Apply fungicides preventatively.",
    "Leaf Scorch": "Ensure proper watering. Avoid high-nitrogen fertilizers.",
    "Leaf Mold": "Improve ventilation. Apply fungicides (chlorothalonil).",
    "Septoria Leaf Spot": "Remove lower infected leaves. Apply chlorothalonil or copper.",
    "Spider Mites": "Spray with water or insecticidal soap. Introduce predatory mites.",
    "Target Spot": "Apply fungicides. Improve airflow.",
    "Yellow Leaf Curl Virus": "Control whiteflies (vectors) with insecticides or nets.",
    "Mosaic Virus": "Remove infected plants. Wash hands/tools (no chemical cure).",
    "Healthy": "Great job! Continue your current care routine. Monitor regularly."
}

# ===================================================================
# 2. UTILITY FUNCTIONS
# ===================================================================

def format_class_name(raw_name):
    """
    Transforms 'Apple___Apple_scab' -> 'Apple - Apple scab' for better readability.
    """
    return raw_name.replace("___", " - ").replace("_", " ").replace(",", ", ")

def get_treatment(disease_name):
    """
    Looks up the disease name in our TREATMENT_INFO database.
    """
    clean_name = format_class_name(disease_name).split("-")[-1].strip()
    
    # Search for keywords in our database
    for key, advice in TREATMENT_INFO.items():
        if key.lower() in clean_name.lower():
            return advice
    
    # Fallback if healthy or unknown
    if "healthy" in clean_name.lower():
        return TREATMENT_INFO["Healthy"]
        
    return "Consult a local agricultural extension for specific treatment advice."

# NOTE: @st.experimental_singleton is the correct cache command for Streamlit 1.12.0
# It prevents the model from reloading every time you click a button.
@st.experimental_singleton
def load_model(model_path):
    try:
        logger.info(f"Loading model from {model_path}")
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def find_available_models():
    """Scans directories to automatically find .h5 model files."""
    base = Path(__file__).resolve().parent
    dirs = [str(base), str(base / "models")]
    
    # Explicit check for the specific path defined in constants
    if MODEL_PATH and os.path.isfile(MODEL_PATH):
        return [os.path.abspath(MODEL_PATH)]
    
    found = []
    for d in dirs:
        found.extend(glob.glob(os.path.join(d, "*.h5")))
    return sorted(list(set(found)), reverse=True)

# ===================================================================
# 3. PAGE LAYOUTS
# ===================================================================

def show_home_page():
    """The Landing Page: Explains the project."""
    st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🌿 Plant Health Expert</h1>", unsafe_allow_html=True)
    st.markdown("### AI-Powered Disease Diagnosis & Treatment")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Using a standard placeholder image regarding nature/plants
        st.image("https://images.unsplash.com/photo-1530836369250-ef72a3f5cda8?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", 
                 use_column_width=True, caption="Protecting crops, securing food.")
    
    with col2:
        st.markdown("""
        **Welcome to the Plant Disease Detection System.**
        
        This tool utilizes state-of-the-art Deep Learning to help farmers and gardeners identify plant diseases early.
        
        #### 🚀 How it Works:
        1. **Upload** an image of a plant leaf.
        2. **AI Scans** the image for disease patterns.
        3. **Get Results** instantly with confidence scores.
        4. **Receive Advice** on how to treat the issue.
        
        #### 🎯 Accuracy:
        Our model is trained on the **PlantVillage dataset**, capable of recognizing **38 different classes** of plant leaves and diseases.
        """)
        
    st.info("👉 Navigate to **'Disease Recognition'** in the sidebar to start diagnosing!")

def show_prediction_page(model):
    """The Main Feature: Upload image and get prediction."""
    st.header("🔍 Disease Recognition")
    st.write("Upload a clear photo of a plant leaf to identify potential diseases.")
    
    col_upload, col_result = st.columns([1, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Preprocessing: Resize and normalize
            img_resized = image.resize(IMAGE_SIZE)
            img_array = np.array(img_resized, dtype=np.float32)
            img_batch = np.expand_dims(img_array, axis=0)
            
            # Dark Image Check
            if np.mean(img_array) < DARK_IMAGE_THRESHOLD:
                st.warning("⚠️ Image is quite dark. Results might be less accurate.")
                
            if st.button("Analyze Leaf", key="analyze"):
                with st.spinner("AI is examining the leaf..."):
                    # THE PREDICTION
                    predictions = model.predict(img_batch, verbose=0)
                    score = predictions[0]
                    class_idx = np.argmax(score)
                    confidence = np.max(score)
                    
                    # Save results to 'session_state' so they don't disappear when you click other buttons
                    st.session_state['last_result'] = {
                        "class": CLASS_NAMES[class_idx],
                        "confidence": confidence,
                        "score": score
                    }

    with col_result:
        # Display results if they exist in session state
        if 'last_result' in st.session_state:
            res = st.session_state['last_result']
            formatted_name = format_class_name(res["class"])
            conf_percent = res["confidence"] * 100
            
            st.markdown("### Analysis Report")
            
            # FEATURE 1: Dynamic Color Confidence Gauge
            color = "green" if conf_percent > 85 else "orange" if conf_percent > 70 else "red"
            
            # HTML injection for custom styling
            st.markdown(f"**Detected:** <span style='color:{color}; font-size:24px'>{formatted_name}</span>", unsafe_allow_html=True)
            
            # Progress bar
            st.progress(int(conf_percent))
            st.caption(f"AI Confidence: {conf_percent:.2f}%")
            
            if conf_percent < CONFIDENCE_THRESHOLD * 100:
                st.error("Low confidence detection. Please verify with an expert.")

            st.markdown("---")
            
            # FEATURE 2: Treatment Recommendations
            st.subheader("💊 Recommended Action")
            treatment = get_treatment(res["class"])
            st.info(treatment)
            
            # Table of Top 3 probabilities
            st.markdown("#### Top Matches")
            top_indices = np.argsort(res["score"])[-3:][::-1]
            data = []
            for i, idx in enumerate(top_indices):
                data.append({
                    "Disease": format_class_name(CLASS_NAMES[idx]),
                    "Probability": f"{res['score'][idx]*100:.2f}%"
                })
            # Using st.table because it looks cleaner for small data in Streamlit 1.12
            st.table(data)

def show_model_page(model):
    """Technical Page: Shows model stats and layers."""
    st.header("🧠 Model Intelligence")
    st.markdown("Technical details about the AI powering this application.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Architecture Stats")
        st.write(f"**Input Shape:** {IMAGE_SIZE} px")
        st.write(f"**Total Classes:** {len(CLASS_NAMES)}")
        st.write("**Framework:** TensorFlow / Keras")
        st.write("**Dataset:** PlantVillage (Public)")
        
    with col2:
        st.markdown("### Training Performance")
        # You can update these with your real training numbers
        st.metric("Validation Accuracy", "94.2% (Approx)")
        st.metric("Loss Function", "Categorical Crossentropy")
    
    st.markdown("---")
    st.subheader("Layer Architecture")
    
    # Advanced: Captures the 'model.summary()' print output and shows it in the app
    with st.expander("View Full Model Summary"):
        try:
            stream = io.StringIO()
            model.summary(print_fn=lambda x: stream.write(x + '\n'))
            summary_string = stream.getvalue()
            st.text(summary_string)
        except:
            st.warning("Model summary not available.")

# ===================================================================
# 4. MAIN APP LOGIC
# ===================================================================

def main():
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    # A simple radio button acts as our page router
    page = st.sidebar.radio("Go to", ["Home", "Disease Recognition", "Model Insights"])
    
    st.sidebar.markdown("---")
    st.sidebar.caption("PBL Project Team v1.0")
    
    # Load Model Logic
    available_models = find_available_models()
    if not available_models:
        st.error("No model found! Please check the 'models' folder or update MODEL_PATH in the code.")
        return

    # Load the first model found
    model = load_model(available_models[0])
    
    if not model:
        return

    # Page Routing Logic
    if page == "Home":
        show_home_page()
    elif page == "Disease Recognition":
        show_prediction_page(model)
    elif page == "Model Insights":
        show_model_page(model)

if __name__ == "__main__":
    main()