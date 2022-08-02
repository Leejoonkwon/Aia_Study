import cv2
import numpy as np
import os

path_dir = "D:\study_data\project\\test"
file_list = os.listdir(path_dir)

file_list[0]
len(file_list)
print(len(file_list))

file_name_list = []

for i in range(len(file_list)):
    file_name_list.append(file_list[i].replace(".jpg",""))
print(file_name_list)

print(file_name_list[0])
def Cutting_face_save(image, name):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
        cropped = image[y: y+h, x: x+w]
        resize = cv2.resize(cropped, (250,250))
        # cv2.imshow("crop&resize", resize)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 이미지 저장하기
        cv2.imwrite(f"D:\study_data\project\\test/{name}.jpg", resize)
        
for name in file_name_list:
    img = cv2.imread("D:\study_data\project\\test/"+name+".jpg")
    Cutting_face_save(img, name)
    
