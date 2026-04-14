import gdown
import os
import streamlit as st
import tensorflow as tf
import numpy as np
from keras.utils import load_img, img_to_array

# title
st.title("Waste Classification AI ♻️")

# load model

model_path = "waste_classifier_model.h5"

if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1YYevZI08SPF3eE5gTBkSoeod7rPR1P1P"
    gdown.download(url, model_path, quiet=False)

model = tf.keras.models.load_model(model_path)

# upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # display image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # process image
    img = load_img(uploaded_file, target_size=(150,150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # prediction
    prediction = model.predict(img_array)

    classes = ["dry", "wet"]
    result = classes[np.argmax(prediction)]

    # result
    st.success(f"Prediction: {result}")