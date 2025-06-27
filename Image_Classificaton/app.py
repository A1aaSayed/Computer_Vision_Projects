import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import requests
import random
import tensorflow as tf
import time
import plotly.express as px
from streamlit_lottie import st_lottie
from gtts import gTTS

# ========== Setup ========== #
st.set_page_config(page_title='CIFAR-10 Classifier', layout='wide')
st.sidebar.title('🔍 Navigation')
page = st.sidebar.radio('Go to', ['Home', 'About', 'How It Works'])

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@st.cache_resource
def download_and_load_model():
    model_path = 'best_cifar10_model.h5'
    if not os.path.exists(model_path):
        st.error("Model file not found. Please upload or download 'efficientNet_cifar10.h5'.")
        st.stop()
    return load_model(model_path)

model = download_and_load_model()

# ========== Helper Functions ========== #
def preprocess(img):
    image = img.resize((32, 32))
    array = np.expand_dims(np.array(image) / 255.0, axis=0)
    return array

def predict_label(image):
    pred = model.predict(preprocess(image), verbose=0)
    confidence = np.max(pred)
    label_index = np.argmax(pred)
    label = class_names[label_index]
    return label, confidence, label_index

def speak(text):
    tts = gTTS(text)
    tts.save('tts.mp3')
    st.audio('tts.mp3')

# ========== Pages ========== #
if page == 'Home':
    st.markdown("<h1 style='text-align: center;'>🧠 CIFAR-10 Image Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Upload an image and let AI guess its category!</h4>", unsafe_allow_html=True)
    st.markdown('---')

    uploaded = st.file_uploader('📤 Upload an image', type=['png', 'jpg', 'jpeg'])
    guess = st.radio('🤔 What do **you** think it is?', ['Not Sure'] + class_names)

    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        st.image(img, caption='🖼️ Uploaded Image')

        with st.spinner('🧠 Analyzing...'):
            label, conf, label_index = predict_label(img)

        emoji_dict = {
            'cat': '🐱', 'dog': '🐶', 'airplane': '✈️', 'automobile': '🚗', 'bird': '🐦', 'deer': "🦌", 'frog': "🐸", 'horse': "🐴", 'ship': "🚢", 'truck': "🚚"
        }
        emoji = emoji_dict.get(label, '📦')

        st.success(f"{emoji} It's a **{label.upper()}** with {conf*100:.2f}% confidence!")

        if guess.lower() == label.lower() and guess != 'Not Sure':
            st.balloons()
            audio_path = f'Deployment/{label}.mp3'
            st.audio(audio_path, format='audio/mp3')
            st.success('🎉 Great job! You guessed it right!')
        elif guess != 'Not Sure':
            st.warning('❗ Not quite. Try again!')

        if st.toggle('🔈 Hear it'):
            speak(f"It's a {label} with {conf*100:.2f} percent confidence.")

        fun_facts = {
            "cat": ["Cats sleep for 13–16 hours a day."],
            "dog": ["Dogs’ noses are as unique as human fingerprints."],
            "airplane": ["The first commercial flight was in 1914."],
            "automobile": ["The first car was made in 1886."],
            "bird": ["Birds are the only animals with feathers."],
            "deer": ["Male deer grow new antlers every year."],
            "frog": ["Frogs absorb water through their skin."],
            "horse": ["Horses can sleep standing up."],
            "ship": ["Ships carry around 90% of world trade."],
            "truck": ["The heaviest truck ever built weighed 360 tons."]
        }
        selected_fact = random.choice(fun_facts[label])
        st.markdown(f"""
        <div style='background-color: #f9f9f9; color: #000000; border-left: 5px solid #f39c12; padding: 1rem; margin-top: 2rem; border-radius: 8px; font-size: 1.1rem;'>
                🧠 <strong>Did you know?</strong><br>{selected_fact}
            </div>
        """, unsafe_allow_html=True)

elif page == 'About':
    st.markdown('## 📘 About This App')
    st.write('''
        This web application uses an **EfficientNetB0** convolutional neural network 
        fine-tuned on the **CIFAR-10** dataset to classify images into 10 common object categories.
        
        It combines deep learning with an interactive interface built using **Streamlit**.
    ''')
    st.image("https://nvsyashwanth.github.io/machinelearningmaster/assets/images/cifar10/cifar10.png", caption="CIFAR-10 Dataset Classes")
    st.write("🔧 Developed by **Alaa Sayed** using TensorFlow and Streamlit.")

elif page == 'How It Works':
    st.markdown('## ⚙️ How It Works')
    st.markdown('''
    Here's what happens behind the scenes:
    - 📷 The uploaded image is resized to **224x224 pixels**
    - 🎨 Pixel values are normalized to the [0, 1] range
    - 🧠 The image is passed through **EfficientNetB0** (pretrained on ImageNet and fine-tuned on CIFAR-10)
    - 📊 The model outputs probability scores for each of the 10 classes
    - ✅ The class with the highest score is returned as the final prediction
    ''')
    st.image("https://www.researchgate.net/profile/Huy-Pham-6/publication/337790072/figure/fig3/AS:843174067056640@1578039771188/Example-of-a-CNN-for-the-Image-Classification-task-on-CIFAR10-dataset.ppm", caption="CIFAR-10 Classifier Architecture")