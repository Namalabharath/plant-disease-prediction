
import os
import json
from PIL import Image, ImageEnhance
import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st

# Configure page
st.set_page_config(
    page_title="🌱 Plant Disease Detection & Treatment",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .remedy-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .sidebar-info {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"

# Load the pre-trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

# Load class names and remedies
@st.cache_data
def load_data():
    class_indices = json.load(open(f"{working_dir}/class_indices.json"))
    remedies = json.load(open(f"{working_dir}/remedies.json"))
    return class_indices, remedies

model = load_model()
class_indices, remedies = load_data()

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    
    return predicted_class_name

# Function to enhance image
def enhance_image(image, brightness=1.0, contrast=1.0, saturation=1.0):
    # Brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    
    # Contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    
    # Saturation
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(saturation)
    
    return image

# Streamlit App
st.markdown('<h1 class="main-header">🌱 Plant Disease Detection & Treatment Assistant</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🔧 Settings & Information")
    
    # Model information
    st.markdown("""
    <div class="sidebar-info">
        <h4>📊 Model Information</h4>
        <ul>
            <li><strong>Model Type:</strong> CNN Deep Learning</li>
            <li><strong>Classes:</strong> 38 Disease Types</li>
            <li><strong>Input Size:</strong> 224x224 pixels</li>
            <li><strong>Accuracy:</strong> ~95%</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Instructions
    st.markdown("""
    <div class="sidebar-info">
        <h4>📝 How to Use</h4>
        <ol>
            <li>Upload a clear image of the plant leaf</li>
            <li>Adjust image settings if needed</li>
            <li>Click 'Analyze Plant' button</li>
            <li>View results and treatment recommendations</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Support information
    st.markdown("""
    <div class="sidebar-info">
        <h4>🌿 Supported Plants</h4>
        <p><strong>Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato</strong></p>
    </div>
    """, unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📸 Upload Plant Image")
    uploaded_image = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a plant leaf for disease detection"
    )
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        
        # Image enhancement controls
        st.markdown("#### 🎨 Image Enhancement")
        col_bright, col_contrast, col_sat = st.columns(3)
        
        with col_bright:
            brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
        with col_contrast:
            contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
        with col_sat:
            saturation = st.slider("Saturation", 0.5, 2.0, 1.0, 0.1)
        
        # Apply enhancements
        enhanced_image = enhance_image(image, brightness, contrast, saturation)
        
        # Display enhanced image
        st.markdown("#### 📷 Enhanced Image Preview")
        st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)
        
        # Analysis button
        analyze_button = st.button(
            "🔬 Analyze Plant Disease", 
            type="primary",
            use_container_width=True
        )

with col2:
    if uploaded_image is not None and analyze_button:
        st.markdown("### 🔬 Analysis Results")
        
        with st.spinner("🧠 Analyzing plant disease..."):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            
        # Display main prediction
        st.markdown(f"""
        <div class="prediction-card">
            <h3>🎯 Prediction Result</h3>
            <h2>{prediction.replace('___', ' - ').replace('_', ' ').title()}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Search for the corresponding remedy details
        remedy = next((item for item in remedies if item["disease"] == prediction), None)
        
        if remedy:
            st.markdown("### 💊 Treatment & Care Recommendations")
            
            # Create tabs for different information
            tab1, tab2, tab3, tab4 = st.tabs(["💊 Treatment", "🌱 Prevention", "🌾 Growing Conditions", "📋 Summary"])
            
            with tab1:
                col_med, col_fert = st.columns(2)
                
                with col_med:
                    st.markdown("#### 💉 Recommended Medicines")
                    if remedy.get("medicines"):
                        for i, medicine in enumerate(remedy["medicines"], 1):
                            st.markdown(f"**{i}.** {medicine}")
                    else:
                        st.success("✅ This plant appears healthy! No treatment needed.")
                
                with col_fert:
                    st.markdown("#### 🌿 Fertilizers")
                    if remedy.get("fertilizers"):
                        for i, fertilizer in enumerate(remedy["fertilizers"], 1):
                            st.markdown(f"**{i}.** {fertilizer}")
            
            with tab2:
                st.markdown("#### 🛡️ Prevention Methods")
                st.info(remedy.get("prevention", "No specific prevention information available."))
            
            with tab3:
                col_soil, col_temp = st.columns(2)
                
                with col_soil:
                    st.markdown("#### 🏔️ Optimal Soil Type")
                    st.success(remedy.get("soil_type", "N/A"))
                
                with col_temp:
                    st.markdown("#### 🌡️ Temperature Range")
                    st.success(remedy.get("temperature_range", "N/A"))
            
            with tab4:
                st.markdown(f"""
                <div class="remedy-card">
                    <h4>📋 Complete Care Summary</h4>
                    <p><strong>🦠 Disease:</strong> {prediction.replace('___', ' - ').replace('_', ' ').title()}</p>
                    <p><strong>💊 Medicines:</strong> {', '.join(remedy.get('medicines', ['None'])) if remedy.get('medicines') else 'Healthy plant - no treatment needed'}</p>
                    <p><strong>🌿 Fertilizers:</strong> {', '.join(remedy.get('fertilizers', ['None']))}</p>
                    <p><strong>🏔️ Soil:</strong> {remedy.get('soil_type', 'N/A')}</p>
                    <p><strong>🌡️ Temperature:</strong> {remedy.get('temperature_range', 'N/A')}</p>
                    <p><strong>🛡️ Prevention:</strong> {remedy.get('prevention', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Download report button
                report_data = {
                    'Disease': prediction.replace('___', ' - ').replace('_', ' ').title(),
                    'Medicines': ', '.join(remedy.get('medicines', ['None'])) if remedy.get('medicines') else 'Healthy plant',
                    'Fertilizers': ', '.join(remedy.get('fertilizers', ['None'])),
                    'Soil Type': remedy.get('soil_type', 'N/A'),
                    'Temperature Range': remedy.get('temperature_range', 'N/A'),
                    'Prevention': remedy.get('prevention', 'N/A')
                }
                
                st.download_button(
                    label="📄 Download Report",
                    data=json.dumps(report_data, indent=2),
                    file_name=f"plant_analysis_{prediction}.json",
                    mime="application/json"
                )
        else:
            st.error("❌ No treatment information available for the predicted disease.")
    
    elif uploaded_image is None:
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h3>📸 Upload an image to get started!</h3>
            <p>Select a clear image of a plant leaf from the left panel to begin analysis.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🌱 Plant Disease Detection & Treatment Assistant | Powered by Deep Learning CNN Model</p>
    <p>⚠️ This tool provides recommendations for educational purposes. Always consult agricultural experts for professional advice.</p>
</div>
""", unsafe_allow_html=True)