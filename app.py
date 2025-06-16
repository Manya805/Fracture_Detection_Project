import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("fracture_detection_model.h5")

# Configure page
st.set_page_config(
    page_title="Bone Fracture Detection",
    page_icon="🦴",
    layout="centered"
)

# --- Title and Intro ---
st.markdown("""
    <h1 style='text-align: center;'>🦴 Bone Fracture Detector</h1>
    <p style='text-align: center; font-size: 18px;'>Upload an X-ray and click the button to check for fractures </p>
    <hr style="margin-top: 10px;">
""", unsafe_allow_html=True)

# --- Step 1: Upload Image ---
st.markdown("### 📤 Step 1: Upload Your X-ray")
uploaded_file = st.file_uploader("Drop an image file here or browse (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')

    st.image(img, caption="📷 Uploaded Image", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔍 Step 2: Detect Fracture")

    if st.button("🚦 Detect Fracture"):
        with st.spinner("Analyzing X-ray..."):
            # Preprocess image
            img_resized = img.resize((128, 128))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Predict
            prediction = model.predict(img_array)[0][0]
            st.markdown("### 🧠 AI Prediction Result")
            if prediction >= 0.5:
                st.success("✅ No Fracture Detected")
                st.progress(min(int(prediction * 100), 100))
                st.metric("Confidence", f"{prediction:.2%}")
            else:
                st.error("❌ Fracture Detected")
                st.progress(min(int((1 - prediction) * 100), 100))
                st.metric("Confidence", f"{(1 - prediction):.2%}")
else:
    st.info("Please upload an X-ray image to begin.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
#st.markdown("<p style='text-align: center; color: gray;'>Made with ❤️ using TensorFlow & Streamlit</p>", unsafe_allow_html=True)

