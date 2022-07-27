#   개인 프로젝트
#   이미지만 보고 감정 파악하기
#   데이터 프레임
#   (데이터 1.)이미지 파악으로 감정 분석 7가지로 분류 
#   (데이터 2.)노래 제목 및 가사 분석 7가지로 분류 

#   모델 
#   (데이터 1.)이미지 데이터 
#   ->ImageDataGenerator로 리스케일 및 증폭 후 npy로 만들기
#   (데이터 2.)노래 제목 및 가사 텍스트 데이터 
#   ->Tokenizer로 수치화 및  npy로 만들기
#   2가지 데이터 프레임 각각 앙상블 후 훈련 

#   평가,예측 
#   PPT 2가지 Time table,주제 선정이유

import numpy as np      
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

#1. 데이터

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest')
                 
test_datagen = ImageDataGenerator(
    rescale=1./255)

xy_train= test_datagen.flow_from_directory(
    'D:\study_data\\train',
    target_size=(150,150),
    class_mode='categorical',
    color_mode='grayscale',
    batch_size=28872,
    shuffle=True,) # 경로 및 폴더 설정

x_train = xy_train[0][0]
y_train = xy_train[0][1]

augument_size = 10000
randidx = np.random.randint(x_train.shape[0],size=augument_size)

x_augumented = x_train[randidx].copy()
y_augumented = y_train[randidx].copy()
x_train = x_train[randidx]
y_train = y_train[randidx]
print(x_augumented.shape) #(25000, 150, 150, 1)
print(y_augumented.shape) #(25000, 21)

xy_df3 = train_datagen.flow(x_augumented,y_augumented,
                       batch_size=augument_size,
                       shuffle=False)
x_data = np.concatenate((x_train,xy_df3[0][0]))
y_data = np.concatenate((y_train,xy_df3[0][1]))


x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,train_size = 0.75,shuffle=False)

np.save('D:\study_data\_save\_npy\_train_x2.npy',arr=x_train)
np.save('D:\study_data\_save\_npy\_train_y2.npy',arr=y_train)
np.save('D:\study_data\_save\_npy\_test_x2.npy',arr=x_test)
np.save('D:\study_data\_save\_npy\_test_y2.npy',arr=y_test)







