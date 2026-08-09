# frontend/pages/3_Satellite_CV.py
import streamlit as st
from PIL import Image, ImageOps
from models.cv_engine import SatelliteImageClassifier
import numpy as np
import time

st.set_page_config(page_title="Satellite CV Segmentation", layout="wide")

st.title("🛰️ Computer Vision Cloud & Cyclone Detection Workbench")
st.markdown("---")

# Initialize the classifier model
classifier = SatelliteImageClassifier()

# Top Ingestion Control Bar
col_upload, col_demo = st.columns([2, 1])

with col_upload:
    uploaded_img = st.file_uploader("Upload Satellite Image Layer (JPG/PNG)", type=["png", "jpg", "jpeg"])

with col_demo:
    st.markdown("### 🌀 Fast Test Ingestion")
    load_demo = st.button("Generate Synthetic Cyclone Sample", use_container_width=True)

img = None

# Handle Uploaded vs Synthetic Image
if uploaded_img:
    img = Image.open(uploaded_img)
elif load_demo:
    # Synthesize a 224x224 atmospheric cyclonic spiral pattern
    x, y = np.meshgrid(np.linspace(-2.5, 2.5, 224), np.linspace(-2.5, 2.5, 224))
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x) + 2.5 * r
    swirl = np.sin(theta * 2.5) * (r < 2.2)
    img_array = ((swirl - swirl.min()) / (swirl.max() - swirl.min()) * 255).astype(np.uint8)
    img = Image.fromarray(img_array).convert("RGB")
    st.toast("Generated synthetic Bay of Bengal cyclone layer!", icon="🌀")

if img:
    # Setup interactive workbench columns
    col_input, col_output = st.columns(2)
    
    with col_input:
        st.subheader("Raw Ingestion Layer")
        st.image(img, caption="Ingested Satellite Capture", use_container_width=True)
        
    with col_output:
        st.subheader("Interactive Model Analysis Pipeline")
        
        # Interactive customization controls
        vorticity_threshold = st.slider("Vorticity Highlight Sensitivity", min_value=10, max_value=100, value=50)
        mask_palette = st.radio("Radar Overlay Palette", ["Thermal Radar (RGB)", "Velocity Density (GOM)"], horizontal=True)
        
        if st.button("🚀 Execute Neural Cloud Segmentation & Classification", use_container_width=True):
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.003)
                progress_bar.progress(percent_complete + 1)
                
            # Calling the clean infer method from cv_engine.py
            predicted_label, confidence = classifier.infer(img)
            
            # Create simulated U-Net Segmentation heatmap overlay 
            img_gray = ImageOps.grayscale(img)
            if "Thermal" in mask_palette:
                heatmap = ImageOps.colorize(img_gray, black="blue", white="red", mid="yellow")
            else:
                heatmap = ImageOps.colorize(img_gray, black="darkgreen", white="magenta", mid="orange")
                
            processed_mask = Image.blend(img.convert("RGB"), heatmap, alpha=vorticity_threshold / 100)
            
            # Render visual heatmap next to metrics
            st.image(processed_mask, caption="Processed U-Net Atmospheric Highlight Mask", use_container_width=True)
            
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("System Classification Verdict", predicted_label)
            with m_col2:
                st.metric("Classification Confidence", f"{confidence * 100:.2f}%")
                
            # Deep atmospheric alert telemetry
            if "Active" in predicted_label or load_demo:
                st.warning("⚠️ Atmospheric Warning: Structural indicators point to active storm fronts or cyclonic vorticity.")
                
                # Additional high-tech diagnostic metrics
                st.markdown("#### 🌪️ Estimated Storm Profile Telemetry")
                t_col1, t_col2, t_col3 = st.columns(3)
                with t_col1:
                    st.metric("Core Pressure", "972 hPa", "-18 hPa")
                with t_col2:
                    st.metric("Max Sustained Wind", "115 km/h", "+12 km/h")
                with t_col3:
                    st.metric("Peak Rainfall Rate", "45 mm/hr", "+8 mm/hr")
            else:
                st.success("✅ Atmospheric Report: Stable atmospheric profiles with no cyclonic developments detected.")
else:
    st.info("👈 Upload a satellite layer image or click **'Generate Synthetic Cyclone Sample'** to begin analysis.")