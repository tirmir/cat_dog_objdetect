from flask import Flask ,request
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import pickle
import cv2
import os
import numpy as np


app = Flask(__name__)

model = load_model('/content/output/detector.h5')
lb = pickle.loads(open('/content/output/lb.pickle', "rb").read())




def preprocess(imagePaths):
    image = load_img(imagePaths, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    return image

def predict(model, path):
    x = preprocess(path)
    boxPreds, labelPreds = model.predict(x)

    return boxPreds, labelPreds

@app.route('/predict', methods = ['POST'])
def home():
    image_path = request.json['image_path']
    box, label = predict(model, image_path)
    conf = float(np.max(label))
    i = np.argmax(label, axis = 1)
    label = lb.classes_[i][0]
    

    return {"label" : label, "confidence" : conf}

if __name__ == '__main__':
    app.run(debug=True)