import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io
import os

st.set_page_config(page_title='Image Filters App', layout='centered')
st.title('🖼️ Image Filters App with OpenCV')

uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    # Convert the image to OpenCV format
    image = Image.open(uploaded_image)
    img_array = np.array(image)
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    st.image(image, caption='Original Image', use_column_width=True)

    st.header('Choose a Filter')
    filter_type = st.selectbox('Select Filter', ['None', 'Grayscale', 'Blur', 'Edge Detection', 'Invert Colors'])

    processed_image = img_cv2.copy()

    if filter_type == 'Grayscale':
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    elif filter_type == 'Blur':
        ksize = st.slider('Kernel Size for Blur', 1, 15, 5, step=2)
        processed_image = cv2.GaussianBlur(processed_image, (ksize, ksize), 0)
    elif filter_type == 'Edge Detection':
        threshold1 = st.slider('Threshold1', 0, 500, 100)
        threshold2 = st.slider('Threshold2', 0, 500, 200)
        processed_image = cv2.Canny(processed_image, threshold1, threshold2)
    elif filter_type == 'Invert Colors':
        processed_image = cv2.bitwise_not(processed_image)

    st.header('Processed Image')
    if filter_type == 'Grayscale' or filter_type == 'Edge Detection':
        st.image(processed_image, use_column_width=True, channels='GRAY')
    else:
        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), use_column_width=True)
    if st.button('Download Processed Image'):
        original_filename = uploaded_image.name
        base_filename = os.path.splitext(original_filename)[0]
    
        if filter_type == "Grayscale" or filter_type == "Edge Detection":
            result = Image.fromarray(processed_image)
        else:
            result = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

        buf = io.BytesIO()
        result.save(buf, format='PNG')
        byte_im = buf.getvalue()

        download_filename = f"{base_filename}_processed_image.png"

        st.download_button(
            label='📥 Download',
            data=byte_im,
            file_name='processed_image.png',
            mime="image/png"
        )
