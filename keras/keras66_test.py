import cv2
import numpy as np
import os

path_dir = "C:\LFW-emotion-dataset\data\LFW-FER\LFW-FER\\train\image"
file_list = os.listdir(path_dir)

file_list[0]
len(file_list)
print(len(file_list))

file_name_list = []

for i in range(len(file_list)):
    file_name_list.append(file_list[i].replace(".jpg",""))
    
print(file_name_list[0])

image = cv2.imread('images/faces/Aaron_Eckhart_0001.jpg')
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
    cv2.imshow("face_recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    