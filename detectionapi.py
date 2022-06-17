from flask import Flask ,request
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2
import os
import numpy as np


app = Flask(__name__)

model = load_model('detector.h5')
lb = pickle.loads(open('lb.pickle', "rb").read())


def preprocess(imagePaths):
    image = load_img(imagePaths, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    return image

def predict(model, imagePath):
    image = preprocess(imagePath)
    bbox, label = model.predict(image)
    X1, Y1, X2, Y2 = bbox[0]
    img = cv2.imread(imagePath)
    img = imutils.resize(img, width = 600)
    h, w = image.shape[:2]	
    X1 = int(X1 * w)
    Y1 = int(Y1 * h)	
    X2 = int(X2 * w)
    Y2 = int(Y2 * h)	
    return ((X1, Y1), (X2, Y2)), label

@app.route('/predict', methods = ['POST'])
def home():
    image_path = request.json['image_path']
    bbox, label = predict(model, image_path)
    conf = float(np.max(label))
    i = np.argmax(label, axis = 1)
    label = lb.classes_[i][0]
    

    return {"label" : label, "confidence" : conf, "bbox" : bbox}

if __name__ == '__main__':
    app.run(debug=True)
