import skimage
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import time
import cv2
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.feature import canny
from skimage import img_as_float64


# def feature_ext(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = canny(img, sigma=1)
#     img = img_as_float64(img)
#     img = np.expand_dims(img, axis=2)
#     return img


# Generating logs using tensorboard
LOG_DIR_NAME = f'BRAIN-TUMOR-PRED-CNN-TL-RESNET-{int(time.time())}'
logs = TensorBoard(log_dir=f'logs\\{LOG_DIR_NAME}\\')


TRAIN_PATH = '../../data/Brain_Tumor_Training'
# Image Augmentation using ImageDataGenerator
train_batches = ImageDataGenerator(
    rescale=1./255,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.1,
)

# Creating training data
train_data = train_batches.flow_from_directory(
    directory=TRAIN_PATH,
    target_size=(224, 224),
    subset='training'
)

# Creating validation data
valid_data = train_batches.flow_from_directory(
    directory=TRAIN_PATH,
    target_size=(224, 224),
    subset='validation'
)

RESNET_MODEL_URL = "https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5"


# Creating a keras layer from the ImageNet model, which will be the first layer of our model (trainable = False means
# we are freezing the model and all the pre-trained layers will not be trained again)
resnet_model = hub.KerasLayer(
    RESNET_MODEL_URL, input_shape=(224, 224, 3), trainable=False
)

# Creating our model by adding some extra layers.
model = tf.keras.Sequential([
    resnet_model,
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax'),
])

print(model.summary())

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=50, batch_size=32, validation_data=valid_data, callbacks=[logs])

model.save('model_brain_tumor_pred_cnn_tl_resnet')
