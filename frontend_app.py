import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io

# --- CONFIGURATION ---
# Point this to your AWS EC2 Nginx Server
API_URL = "http://51.21.246.252/predict"

st.set_page_config(layout="centered", page_title="ElectroVision Cloud")

# --- VISUAL PROCESSING FUNCTIONS (UI ONLY) ---
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

def get_grabcut_mask(frame, rect):
    try:
        mask = np.zeros(frame.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        binary_mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 255).astype('uint8')
        return binary_mask
    except:
        return None

def apply_grabcut_blur(frame, mask, blur_kernel_size=21):
    k_size = blur_kernel_size if blur_kernel_size % 2 != 0 else blur_kernel_size + 1
    background_blur = cv2.GaussianBlur(frame, (k_size, k_size), 0)
    
    soft_mask = cv2.GaussianBlur(mask, (11, 11), 0).astype(np.float32) / 255.0
    soft_mask_3ch = cv2.cvtColor(soft_mask, cv2.COLOR_GRAY2BGR)
    
    foreground = frame.astype(np.float32) * soft_mask_3ch
    background = background_blur.astype(np.float32) * (1.0 - soft_mask_3ch)
    return cv2.add(foreground, background).astype("uint8")

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

uploaded_file = None
if option == "Camera":
    uploaded_file = st.camera_input("Take a picture of an electronic accessory")
else:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display original image and convert for UI processing
    pil_image = Image.open(uploaded_file).convert("RGB")
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # 2. Convert to bytes to send over the internet
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    with st.spinner('Sending to AWS Cloud for Analysis...'):
        try:
            # --- THE MAGIC: CALLING YOUR API ---
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
                    # Apply UI Visual Effects (GrabCut & Lighting)
                    h, w, _ = frame.shape
                    rect_dim = int(min(h, w) * 0.90)
                    rect = ((w - rect_dim) // 2, (h - rect_dim) // 2, rect_dim, rect_dim)
                    
                    mask = get_grabcut_mask(frame, rect)
                    if mask is not None:
                        frame = apply_grabcut_blur(frame, mask)
                    
                    frame = apply_sharpen(apply_studio_lighting(frame))
                    
                    # Display Results
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Processed by UI", use_column_width=True)
                    st.success(f"### Predicted Class: **{class_name}**")
                    st.info(f"### Cloud Confidence: **{confidence:.2f}%**")
            else:
                st.error(f"AWS Server Error: {response.text}")

        except Exception as e:
            st.error(f"Failed to connect to AWS backend. Is the server running? Error: {e}")