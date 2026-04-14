import tensorflow as tf
import numpy as np
from keras.utils import load_img, img_to_array

# load model
model = tf.keras.models.load_model("waste_classifier_model.h5")

# load image
img = load_img("test.jpg", target_size=(150,150))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# prediction
prediction = model.predict(img_array)

# class labels
classes = ["dry", "wet"]

# result
result = classes[np.argmax(prediction)]

print("Prediction:", result)