import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import csv
import cv2
import numpy as np

import numpy
#load the trained model to classify sign
from keras.models import load_model
model = load_model('my_model.h5')

#dictionary to label all traffic signs class.
dic = {}
with open('./archive/labels.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            dic[int(row[0])] = row[1]

def classify(file_path,dic):
    image = Image.open(file_path)
    image = image.resize((30,30))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    print(image.shape)
    pred_probabilities = model.predict(image)[0]
    pred = pred_probabilities.argmax(axis=-1)
    sign = dic[pred]
    return sign
   
def upload_image(file_path):
    try:
        img = cv2.imread(file_path)
        img = cv2.resize(img, (400,400))
        sign = classify(file_path, dic)
        print(sign)
        cv2.imshow('img_denoise', img)
        cv2.waitKey(0)
    except:
        pass

upload_image('./archive/traffic_Data/TEST/008_0001_j.png')
# upload_image('./data/stop-signs-eight-sides.jpg')