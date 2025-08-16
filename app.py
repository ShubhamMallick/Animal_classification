import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
import io

# Load model
model = tf.keras.models.load_model("animal_classifier_model.h5")

# Load class indices
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}

# Set up the Streamlit page
st.set_page_config(page_title="Animal Classifier 🐾", layout="centered")
st.title("🐾 Animal Image Classifier")
st.markdown("Upload an animal image and let the model tell you which animal it is!")

# Show instructions
with st.expander("📌 Instructions", expanded=True):
    st.markdown("""
    This app classifies animal images using a trained deep learning model. Please follow the steps below:

    1. **Upload a clear image** of a single animal.
    2. Supported image formats: **JPG, JPEG, PNG**.
    3. The image will be resized to **224x224** before classification.
    4. Model will predict from the following 15 animals:
    """)
    
    animal_list = [
        "🐻 Bear", "🐦 Bird", "🐱 Cat", "🐄 Cow", "🦌 Deer",
        "🐶 Dog", "🐬 Dolphin", "🐘 Elephant", "🦒 Giraffe",
        "🐴 Horse", "🦘 Kangaroo", "🦁 Lion", "🐼 Panda",
        "🐅 Tiger", "🦓 Zebra"
    ]
    
    st.markdown("### 🔍 Classes:")
    st.markdown(", ".join(animal_list))

# Upload image
uploaded_file = st.file_uploader("📁 Choose an animal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📷 Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    pred = model.predict(img_array)
    pred_index = np.argmax(pred)
    pred_class = index_to_class[pred_index]
    confidence = float(np.max(pred)) * 100

    # Annotate prediction on image
    annotated_img = img.copy()
    draw = ImageDraw.Draw(annotated_img)
    font = ImageFont.load_default()
    text = f"{pred_class} ({confidence:.2f}%)"
    draw.text((10, 10), text, fill=(255, 0, 0), font=font)

    # Display results
    st.markdown(f"### ✅ Prediction: **{pred_class}**")
    st.markdown(f"### 📊 Confidence: **{confidence:.2f}%**")
    st.image(annotated_img, caption="🧠 Prediction Result", use_column_width=True)
