# 🖼️ Image Filters App

A simple interactive **web application** built using **Streamlit** and **OpenCV** that allows you to upload an image and apply various filters to it in real time.

After processing, you can download the modified image with a filename that includes `_processed_image`.

---

## 🎨 Features

- ✅ Upload `.jpg`, `.jpeg`, or `.png` images
- ✅ Apply filters:
  - Grayscale
  - Blur
  - Edge Detection (Canny)
  - Invert Colors
  - Solarize
  - Dust and Scratches
  - Emboss
- ✅ Preview original and processed images
- ✅ Download the processed image

---

## 🚀 Demo

Here's a short demo of the app in action:

![Image Filters App Demo](assets/demo.gif)

---

## 📦 Installation & Usage

### 1️⃣ Clone the repository

```bash
git clone https://github.com/A1aaSayed/Computer_Vision_Projects.git
cd Computer_Vision_Projects/Image_Filters_App
```

---

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app
```bash
streamlit run image_filters_app.py
```