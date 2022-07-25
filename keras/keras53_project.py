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

#1. 데이터

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest')
                 
test_datagen = ImageDataGenerator(
    rescale=1./255)

xy_train= train_datagen.flow_from_directory(
    'D:\study_data\_data\emotion\\train',
    target_size=(150,150),
    class_mode='categorical',
    batch_size=28709,
    shuffle=True,) # 경로 및 폴더 설정
xy_test= test_datagen.flow_from_directory(
    'D:\study_data\_data\emotion\\test',
    target_size=(150,150),
    class_mode='categorical',
    batch_size=7178,
    shuffle=True,) # 경로 및 폴더 설정
# print(xydata[0][0],xydata[0][0].shape) # (500, 150, 150, 3)
print(xy_train[0][0].shape) # (500, 150, 150, 3)
print(xy_train[0][1].shape) # (500, 7)
x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

augument_size = 500
randidx = np.random.randint(x_train.shape[0],size=augument_size)

x_augumented = x_train[randidx].copy()
y_augumented = y_train[randidx].copy()


print(x_augumented.shape)  #(10000, 150, 150, 3)
print(y_augumented.shape) #(10000, 7)
print(x_train.shape) #(28709, 150, 150, 3)




x_df = np.concatenate((x_train,x_augumented))
y_df = np.concatenate((y_train,y_augumented))
# print(x_df.shape) #(64000, 28, 28, 1)

xy_df3 = test_datagen.flow(x_df,y_df,
                       batch_size=augument_size,shuffle=False)

np.save('D:\study_data\_data\_save\_npy\_train_x.npy',arr=xy_df3[0][0])
np.save('D:\study_data\_data\_save\_npy\_train_y.npy',arr=xy_df3[0][1])
np.save('D:\study_data\_data\_save\_npy\_test_x.npy',arr=x_test)
np.save('D:\study_data\_data\_save\_npy\_test_y.npy',arr=y_test)







