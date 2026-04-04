import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model("waste_classifier.h5")

# Title
st.title("♻️ WasteSort AI")
st.write("Upload an image to classify waste as Recyclable or Non-Recyclable")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Class labels
class_names = ["non_recyclable", "recyclable"]

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=500)

    # Preprocess image
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    confidence = prediction[0][0]

    if confidence > 0.5:
        result = "♻️ Recyclable"
        conf = confidence
    else:
        result = "❌ Non-Recyclable"
        conf = 1 - confidence

    # Display result
    st.subheader(f"Prediction: {result}")
    st.write(f"Confidence: {conf*100:.2f}%")
