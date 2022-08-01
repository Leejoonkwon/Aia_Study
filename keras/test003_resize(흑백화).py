import cv2
import os
import numpy as np
from PIL import Image
 
path = 'D:\study_data\project\\test'
imagePaths = [os.path.join(path,file_name) for file_name in os.listdir(path)]
for imagePath in imagePaths:
    img = Image.open(imagePath).convert('L')
    img_numpy = np.array(img, 'uint8')
    cv2.imwrite("D:\study_data\project\\test\\" + imagePath.split("\\")[-1], img_numpy)
print("All Done")