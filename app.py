import gdown
import os
import streamlit as st
import tensorflow as tf
import numpy as np
from keras.utils import load_img, img_to_array

st.title("Waste Classification AI ♻️")

model_path = "waste_classifier_model.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=1YYevZI08SPF3eE5gTBkSoeod7rPR1P1P"
        gdown.download(url, model_path, quiet=False)

    return tf.keras.models.load_model(model_path)

model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    img = load_img(uploaded_file, target_size=(150,150), color_mode="rgb")
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)

    classes = ["dry", "wet"]
    result = classes[np.argmax(prediction)]

    st.success(f"Prediction: {result}")
