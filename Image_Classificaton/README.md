# 🧠 CIFAR-10 Image Classifier Web Application

This project presents a web-based image classification tool built using **Streamlit** and trained models on the **CIFAR-10** dataset. The app allows users to upload an image and get an AI-powered prediction of the image class, along with a confidence score and voice feedback.

---

## 📌 Overview

This project includes **two deep learning models**:

1. A **custom Convolutional Neural Network (CNN)** built from scratch.
2. A **transfer learning model using EfficientNetV2B0** pretrained on ImageNet.

Both models are trained to classify images into 10 categories from the CIFAR-10 dataset.

---

## 🧰 Tech Stack

- **Frontend:** Streamlit (Python)
- **Model Framework:** TensorFlow / Keras
- **Dataset:** CIFAR-10 (preloaded from Keras Datasets)
- **Audio Feedback:** gTTS (Google Text-to-Speech)
- **Data Visualization:** Plotly, Seaborn, Matplotlib (for analysis)
- **Image Processing:** PIL (Pillow)

---

## 🧪 Model Architectures

### 🔧 1. Custom CNN Model

- Convolutional Layers with ReLU
- Batch Normalization
- Max Pooling & Dropout
- Dense Layers
- Softmax output (10 classes)

Training:
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Epochs: 50
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

---

### ⚙️ 2. EfficientNetV2B0 Transfer Learning

This model uses a pretrained **EfficientNetV2B0** as a feature extractor.

#### Key Details:
- CIFAR-10 images resized from **32x32 → 224x224**
- Feature extraction via **EfficientNetV2B0** (`include_top=False`)
- Added:
  - `GlobalAveragePooling2D`
  - `Dropout`
  - `Dense` layer with softmax for classification

---

## 🚀 Running the App

### Prerequisites

> Ensure the following Python packages are installed:

```bash
pip install streamlit tensorflow pillow gtts plotly
```

## Launch the App
1. Clone this repository or download the files.

2. Ensure the trained model file best_cifar10_model.h5 is located in the same directory as the app.

3. Run the Streamlit application:

```bash
streamlit run app.py
```

## 🖼️ Image Preprocessing
- Input images are resized to 32x32 pixels to match the CIFAR-10 input shape.

- Images are normalized to a [0, 1] pixel intensity range.

- The model processes a single image at inference time using numpy.expand_dims.

## 📚 About CIFAR-10
CIFAR-10 is a well-known benchmark dataset in the field of computer vision. It includes the following classes:

- Airplane ✈️

- Automobile 🚗

- Bird 🐦

- Cat 🐱

- Deer 🦌

- Dog 🐶

- Frog 🐸

- Horse 🐴

- Ship 🚢

- Truck 🚚

Each class is mutually exclusive, and the images are uniformly distributed across categories.
