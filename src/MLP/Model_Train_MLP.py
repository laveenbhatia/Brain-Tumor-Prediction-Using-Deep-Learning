import time
import numpy as np
import random
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator

NAME = f'BRAIN-TUMOR-PRED-MLP-{int(time.time())}'
tensorboard = TensorBoard(log_dir=f'logs\\{NAME}\\')

DIRECTORY = "../../data/Brain_Tumor_Training"
CATEGORIES = ['glioma', 'meningioma', 'notumor', 'pituitary'
              ]
data = []
i = 0

print("Preparing Training Images...")
for category in CATEGORIES:
    folder = os.path.join(DIRECTORY, category)
    label = CATEGORIES.index(category)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr, (128, 128))
        data.append([img_arr, label])
        i = i+1

print(i, "Training images found")

random.shuffle(data)

X = []
y = []

for features, labels in data:
    X.append(features)
    y.append(labels)

X = np.array(X)
y = np.array(y)

X = X.reshape(-1, 49152)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(4, activation='sigmoid')
])

model.build((None, 49152))
print(model.summary())

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=150, validation_split=0.1, batch_size=32, callbacks=[tensorboard])

model.save('model_brain_tumor_pred_mlp')
