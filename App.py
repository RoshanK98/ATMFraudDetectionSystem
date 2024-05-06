from flask import Flask, render_template, request, redirect, url_for, flash, session
#from flaskext.mysql import MySQL
#import pymysql
import re

from tensorflow import keras
from keras.preprocessing import image
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from werkzeug.utils import secure_filename


app = Flask(__name__)

app.secret_key = 'atm-detection'


model = load_model('Detection10EpochsCategorical.h5')
print('Model Loaded')


def get_className(classNo):
    if classNo == 0:
        return "You are fully covered your face with a mask. Please Remove Your mask"
    elif classNo == 1:
        return "You are not covered your face"
    elif classNo == 2:
        return "You are partially covered your face with a mask. Please Remove Your mask"
    else:
        return "You wear a helmet. Please remove"


def getResult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = np.argmax(model.predict(input_img))
    return result


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value = getResult(file_path)
        result = get_className(value)
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

