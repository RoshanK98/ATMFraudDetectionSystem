import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
#from keras.utils.vis_utils import plot_model

# initialized the image path
image_directory = 'datasets/'

# create a list for all the images inside the folders
fully_covered_images = os.listdir(image_directory + 'fully_covered/')
not_covered_images = os.listdir(image_directory + 'not_covered/')
partially_covered_images = os.listdir(image_directory + 'partially_covered/')
with_helmet_images = os.listdir(image_directory + 'with_helmet/')

# to seperate the images & convert to dependent variable
dataset = []
label = []

INPUT_SIZE = 64

for i, image_name in enumerate(fully_covered_images):
    if image_name.split('.')[1] == 'png':
        image = cv2.imread(image_directory + 'fully_covered/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(not_covered_images):
    if image_name.split('.')[1] == 'png':
        image = cv2.imread(image_directory + 'not_covered/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

for i, image_name in enumerate(partially_covered_images):
    if image_name.split('.')[1] == 'png':
        image = cv2.imread(image_directory + 'partially_covered/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(2)


for i, image_name in enumerate(with_helmet_images):
    if image_name.split('.')[1] == 'png':
        image = cv2.imread(image_directory + 'with_helmet/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(3)


dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.2, random_state =0)

# print(x_train.shape)
# print(y_train.shape)

# print(x_test.shape)
# print(y_test.shape)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

y_train = to_categorical(y_train, num_classes=4)
y_test = to_categorical(y_test, num_classes=4)

# model building

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=100, validation_data=(x_test, y_test), shuffle=False)

model.save('Detection10EpochsCategorical.h5')
