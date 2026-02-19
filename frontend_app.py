import streamlit as st
import requests
from PIL import Image
import io

# --- CONFIGURATION ---
# Point this to your AWS EC2 Nginx Server
API_URL = "http://51.21.246.252/predict"

st.set_page_config(layout="centered", page_title="ElectroVision Cloud")

# --- STREAMLIT UI ---
st.title("⚡ ElectroVision Cloud")
st.write("Powered by AWS EC2 & FastAPI")

with st.expander("Photo Guidelines"):
    st.markdown("""
    * **Center the Object:** Ensure the accessory is in the middle.
    * **Clear Background:** Try to have a plain, non-cluttered background.
    * **Good Lighting:** Make sure the object is well-lit.
    """)

# Options for uploading or using camera
option = st.radio("Choose input method:", ("Camera", "Upload Image"))

if option == "Camera":
    uploaded_file = st.camera_input("Take a picture of an electronic accessory")
else:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Read the image directly from the user
    pil_image = Image.open(uploaded_file).convert("RGB")

    # 2. Convert to bytes to send over the internet
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    with st.spinner('Sending to AWS Cloud for Analysis...'):
        try:
            # --- CALLING YOUR API ---
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
                    # --- DISPLAY ORIGINAL IMAGE AND RESULTS ---
                    st.image(pil_image, caption="Captured Image", use_container_width=True)
                    st.success(f"### Predicted Class: **{class_name}**")
                    st.info(f"### Cloud Confidence: **{confidence:.2f}%**")
            else:
                st.error(f"AWS Server Error: {response.text}")

        except Exception as e:
            st.error(f"Failed to connect to AWS backend. Is the server running? Error: {e}")
