import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io

# --- CONFIGURATION ---
API_URL = "http://51.21.246.252/predict"
FEEDBACK_URL = "http://51.21.246.252/feedback"

CLASS_NAMES = [
    'Charger', 'Game Controller', 'Headphone', 'Keyboard', 'Laptop',
    'Monitor', 'Mouse', 'Smartphone', 'Smartwatch', 'Speaker'
]

st.set_page_config(layout="centered", page_title="ElectroVision Cloud")

# --- EDGE COMPUTING FUNCTIONS (PRE-PROCESSING) ---
def apply_studio_lighting(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def apply_sharpen(frame):
    kernel = np.array([[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]])
    return cv2.filter2D(frame, -1, kernel)

def edge_preprocess(pil_image):
    """Resizes and cleans the image locally before sending to AWS."""
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # Resize to reduce network payload size
    frame = cv2.resize(frame, (224, 224))
    
    # Normalize lighting and sharpen
    frame = apply_sharpen(apply_studio_lighting(frame))
    
    # Convert back to PIL for transmission
    processed_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(processed_rgb)

# --- STREAMLIT UI ---
st.title("⚡ ElectroVision Cloud")
st.write("Powered by AWS EC2 & Edge Computing")

with st.expander("Photo Guidelines"):
    st.markdown("""
    * **Center the Object:** Ensure the accessory is in the middle.
    * **Clear Background:** Try to have a plain, non-cluttered background.
    * **Good Lighting:** Make sure the object is well-lit.
    """)

option = st.radio("Choose input method:", ("Camera", "Upload Image"))

uploaded_file = None
if option == "Camera":
    uploaded_file = st.camera_input("Take a picture of an electronic accessory")
else:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_pil = Image.open(uploaded_file).convert("RGB")
    
    # 1. Execute Edge Preprocessing
    processed_pil = edge_preprocess(original_pil)

    # 2. Convert to bytes for transmission
    img_byte_arr = io.BytesIO()
    processed_pil.save(img_byte_arr, format='JPEG', quality=95)
    img_bytes = img_byte_arr.getvalue()

    st.image(processed_pil, caption="Edge Preprocessed Image (Sent to Cloud)", use_column_width=True)

    with st.spinner('Sending to AWS Cloud for Analysis...'):
        try:
            # --- API CALL ---
            files = {'file': ('image.jpg', img_bytes, 'image/jpeg')}
            response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                data = response.json()
                class_name = data["prediction"]
                confidence = data["confidence_percent"]

                if confidence < 50:
                    st.error("### Prediction Inconclusive")
                    st.warning("Please try again with better lighting or a clearer background.")
                else:
                    st.success(f"### Predicted Class: **{class_name}**")
                    st.info(f"### Cloud Confidence: **{confidence:.2f}%**")
                    
                    # --- HUMAN-IN-THE-LOOP FEEDBACK SYSTEM ---
                    st.markdown("---")
                    st.write("### 🧠 Help Improve the Model")
                    st.write("Only submit feedback if the prediction above is **incorrect**.")
                    
                    with st.form("feedback_form"):
                        actual_class = st.selectbox("What is the correct accessory?", ["Select..."] + CLASS_NAMES)
                        submit_feedback = st.form_submit_button("Submit Correction to AWS")

                        if submit_feedback:
                            if actual_class == "Select...":
                                st.warning("Please select the correct class before submitting.")
                            else:
                                feedback_data = {
                                    "predicted_class": class_name,
                                    "actual_class": actual_class,
                                    "confidence": str(confidence)
                                }
                                # Re-create file payload for the feedback endpoint
                                files_fb = {'file': ('error.jpg', img_bytes, 'image/jpeg')}
                                fb_response = requests.post(FEEDBACK_URL, data=feedback_data, files=files_fb)
                                
                                if fb_response.status_code == 200:
                                    st.success("Correction saved to AWS S3 & DynamoDB! Thank you.")
                                else:
                                    st.error(f"Failed to save feedback: {fb_response.text}")

            else:
                st.error(f"AWS Server Error: {response.text}")

        except Exception as e:
            st.error(f"Failed to connect to AWS backend. Is the server running? Error: {e}")
