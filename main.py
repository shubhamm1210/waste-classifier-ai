import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/TRAIN",
    image_size=(150, 150),
    batch_size=16
).take(200)   # 🔥 only 200 batches

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/TEST",
    image_size=(150, 150),
    batch_size=16
).take(50)    # 🔥 smaller test

print("Dataset loaded successfully")
# normalization (image pixel 0–255 → 0–1)
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
test_data = test_data.map(lambda x, y: (normalization_layer(x), y))


# CNN model
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')   # 2 classes
])


# compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# summary (model structure)
model.summary()


# training
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=3
)
model.save("waste_classifier_model.h5")
print("Model saved successfully")