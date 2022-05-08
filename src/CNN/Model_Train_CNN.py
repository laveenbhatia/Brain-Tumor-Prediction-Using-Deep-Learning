import time
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator

NAME = f'BRAIN-TUMOR-PRED-CNN-{int(time.time())}'
tensorboard = TensorBoard(log_dir=f'logs\\{NAME}\\')

TRAIN_PATH = '../../data/Brain_Tumor_Training'
# Image Normalization and Augmentation
train_batches = ImageDataGenerator(
    rescale=1./255,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.1
)
# Creating Training data by converting images to numpy array
train_data = train_batches.flow_from_directory(
    directory=TRAIN_PATH,
    target_size=(128, 128),
    subset='training'
)
# Creating Testing data by converting images to numpy array
valid_data = train_batches.flow_from_directory(
    directory=TRAIN_PATH,
    target_size=(128, 128),
    subset='validation'
)

X, y = next(train_data)

# Creating CNN model using Sequential class
model = Sequential()
# Convolutional layer
model.add(Conv2D(256, (3, 3), activation='relu'))
# Pooling layer
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
# Flatten layer
model.add(Flatten())
# Dense Layers
model.add(Dense(256, input_shape=X.shape[1:], activation='relu'))
model.add(Dense(256, input_shape=X.shape[1:], activation='relu'))
model.add(Dense(128, input_shape=X.shape[1:], activation='relu'))
model.add(Dense(128, input_shape=X.shape[1:], activation='relu'))
model.add(Dense(64, input_shape=X.shape[1:], activation='relu'))
model.add(Dense(64, input_shape=X.shape[1:], activation='relu'))
model.add(Dense(4, activation='softmax'))

model.build((None, 128, 128, 3))
print(model.summary())

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_data,
          epochs=100,
          batch_size=32,
          validation_data=valid_data,
          callbacks=[tensorboard])

# model.save('model_brain_tumor_pred_cnn')
