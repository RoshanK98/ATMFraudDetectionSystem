import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('Detection10EpochsCategorical.h5')

image = cv2.imread('C:\\Users\\USER\\PycharmProjects\\ATM_Detection\\test\\8.png')

img = Image.fromarray(image)

img = img.resize((64, 64))

img = np.array(img)

input_img = np.expand_dims(img, axis=0)

result = np.argmax(model.predict(input_img))
print(result)


