import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json

st.title("🌿 Plant Disease Detection App")

st.warning("⚠️ Please upload only plant leaf images")


# Load model
model = tf.keras.models.load_model("plant_model.h5")


# Load class names
with open("classes.json", "r") as f:
    class_names = json.load(f)

class_names = {v:k for k,v in class_names.items()}

# Upload image
uploaded_file = st.file_uploader("Upload leaf image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image")

    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.reshape(img, (1,224,224,3))

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    result = class_names[class_index]

    st.success(f"Prediction: {result}")
