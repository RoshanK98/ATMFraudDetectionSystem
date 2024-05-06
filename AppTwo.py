from flask import Flask, render_template, request, redirect, url_for, flash, session
#from flaskext.mysql import MySQL
from flask import Flask, render_template, request, redirect, url_for, flash, session
#from flaskext.mysql import MySQL
#import pymysql
import re
import json
from tensorflow import keras
from keras.preprocessing import image
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from werkzeug.utils import secure_filename

import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import glob

import geocoder

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

    # send email if the result matches any of the conditions
    def get_location():
        # get public IP address
        ip_response = requests.get('https://api.ipify.org?format=json')
        ip_address = ip_response.json()['ip']

        # get location info based on IP address
        location_response = requests.get(f'https://ipapi.co/{ip_address}/json/')
        location_data = json.loads(location_response.text)

        city = location_data['city']
        region = location_data['region']
        country = location_data['country_name']
        lat = location_data['latitude']
        lon = location_data['longitude']

        return city, region, country, lat, lon

    result = 2  # assuming result is 2 for testing purposes
    if result == 0 or result == 2 or result == 3:
        city, region, country, lat, lon = get_location()
        maps_url = f'https://www.google.com/maps/search/?api=1&query={lat},{lon}'
        subject = 'SUSPECTED MOMENT DETECTED'
        body = f'The current location of the suspected activity detected is {city}, {region}, {country}. Click the following link to view the location on Google Maps: {maps_url}'
        # send email with subject and body

        gmail_user = 'rk3055089@gmail.com'
        app_password = 'pssz aqhi ooal weho'
        to = 'rk3055089@gmail.com'

        msg = MIMEMultipart()
        msg['From'] = gmail_user
        msg['To'] = to
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # get the latest file from the uploads directory
        list_of_files = glob.glob('uploads/*')
        latest_file = max(list_of_files, key=os.path.getctime)
        with open(latest_file, 'rb') as f:
            attachment = MIMEApplication(f.read(), _subtype="jpg")
            attachment.add_header('Content-Disposition', 'attachment', filename=latest_file)

        msg.attach(attachment)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(gmail_user, app_password)
        text = msg.as_string()
        server.sendmail(gmail_user, to, text)
        server.quit()

    return get_className(result)


def get_location():
    g = geocoder.ip('me')
    city = g.city
    region = g.state
    country = g.country
    lat = g.lat
    lon = g.lng
    return city, region, country, lat, lon


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
        result = getResult(file_path)
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
