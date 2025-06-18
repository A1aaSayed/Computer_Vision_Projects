import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io
import os

st.set_page_config(page_title='Image Filters App', layout='centered')
st.title('🖼️ Image Filters App with OpenCV')

uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

# -------------------- Filters Functions --------------------
def apply_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_blur(img, ksize):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def apply_canny(img, threshold1, threshold2):
    return cv2.Canny(img, threshold1, threshold2)

def apply_invert(img):
    return cv2.bitwise_not(img)

def apply_solarize(img, threshold=128):
    return cv2.bitwise_xor(img, cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1])

def apply_dust_and_scratches(img):
    return cv2.medianBlur(img, 5)

def apply_emboss(img):
    kernel = np.array([[ -2, -1, 0],
                       [ -1, 1, 1],
                       [  0, 1, 2]])
    embossed = cv2.filter2D(img, -1, kernel)
    return embossed

# -------------------- Main App --------------------
if uploaded_image is not None:
    # Convert the image to OpenCV format
    image = Image.open(uploaded_image)
    img_array = np.array(image)
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    st.image(image, caption='Original Image', use_column_width=True)

    st.header('Choose a Filter')
    filter_type = st.selectbox('Select Filter', ['None', 'Grayscale', 'Blur', 'Edge Detection', 'Invert Colors', 'Solarize', 'Dust and Scratches', 'Emboss'])

    processed_image = img_cv2.copy()

    if filter_type == 'Grayscale':
        processed_image = apply_grayscale(processed_image)
    elif filter_type == 'Blur':
        ksize = st.slider('Kernel Size for Blur', 1, 25, 5, step=2)
        processed_image = apply_blur(processed_image, ksize)
    elif filter_type == 'Edge Detection':
        threshold1 = st.slider('Threshold1', 0, 500, 100)
        threshold2 = st.slider('Threshold2', 0, 500, 200)
        processed_image = apply_canny(processed_image, threshold1, threshold2)
    elif filter_type == 'Invert Colors':
        processed_image = apply_invert(processed_image)
    elif filter_type == 'Solarize':
        threshold = st.slider('Solarize Threshold', 0, 255, 128)
        processed_image = apply_solarize(processed_image, threshold)
    elif filter_type == 'Dust and Scratches':
        processed_image = apply_dust_and_scratches(processed_image)
    elif filter_type == 'Emboss':
        processed_image = apply_emboss(processed_image)

    if filter_type != 'None':
        st.header('Processed Image')

    # Display
    if filter_type == 'Grayscale' or filter_type == 'Edge Detection':
        st.image(processed_image, use_column_width=True, channels='GRAY')
        result_image = Image.fromarray(processed_image)
    else:
        rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        st.image(rgb_image, use_column_width=True)
        result_image = Image.fromarray(rgb_image)
        
    # Download processed image
    original_filename = uploaded_image.name
    base_filename = os.path.splitext(original_filename)[0]
    download_filename = f"{base_filename}_{filter_type.lower().replace(' ', '_')}_image.png"
    
    buf = io.BytesIO()
    result_image.save(buf, format='PNG')
    byte_im = buf.getvalue()

    st.download_button(
        label='📥 Download Processed Image',
        data=byte_im,
        file_name=download_filename,
        mime="image/png"
    )