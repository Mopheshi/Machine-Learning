import re
import numpy as np
import cv2
import json
import time
import os

path = "cropped"
files = os.listdir("original")
print(files)

count = 1
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml");
for i in range(len(files)):
    img = cv2.imread("original/" + files[i])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:
        roi_color = img[y:y + h, x:x + w]
        cv2.imwrite(path + "/" + str(count) + ".jpg", roi_color)
        count = count + 1
