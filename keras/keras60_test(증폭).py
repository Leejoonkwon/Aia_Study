from PIL import Image
import os, glob, numpy as np
from sklearn.model_selection import train_test_split
import numpy as np      
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

file_name_freq +=1

data_aug_gen =  ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.5,
    zoom_range=[0.8,2.0],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
img = load_img(fname)
x = img_to_array(img)