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

train = ImageDataGenerator(
    rescale=1./255,               # MinMax 스케일링과 같은 개념 
  )                               # 회전 축소 등으로 이미지에 여백이생겼을때 채우는 방법입니다.
test_datagen = ImageDataGenerator(
    rescale=1./255)

xydata = train.flow_from_directory(
    'C:\study\Study\_data\emotion\\train',
    target_size=(150,150),
    class_mode='categorical',
    batch_size=500,
    shuffle=True,) # 경로 및 폴더 설정

# print(xydata[0][0],xydata[0][0].shape) # (500, 150, 150, 3)
print(xydata[0][0].shape) # (500, 150, 150, 3)
print(xydata[0][1].shape) # (500,)

x_train = xydata[0][0][0:450]
y_train = xydata[0][1][0:450]
x_test = xydata[0][0][450:]
y_test = xydata[0][1][450:]

print(x_train.shape) # (450, 150, 150, 3)
print(y_train.shape) # (450,3)


augument_size = 500
randidx = np.random.randint(x_train.shape[0],size=augument_size)

x_augumented = x_train[randidx].copy()
y_augumented = y_train[randidx].copy()


print(x_augumented.shape)  #(400, 28, 28)
print(y_augumented.shape) #(400,) 50000, 32, 32, 3)
print(x_train.shape) #(160, 150, 150, 1)




xy_df2 = train.flow(x_train,y_train,
                                  batch_size=augument_size,shuffle=False)
x_df = np.concatenate((x_train,x_augumented))
y_df = np.concatenate((y_train,y_augumented))
# print(x_df.shape) #(64000, 28, 28, 1)

xy_df3 = test_datagen.flow(x_df,y_df,
                       batch_size=augument_size,shuffle=False)

np.save('D:/study_data/_save/_npy/keras49_9_train_x.npy',arr=xy_df3[0][0])
np.save('D:/study_data/_save/_npy/keras49_9_train_y.npy',arr=xy_df3[0][1])
np.save('D:/study_data/_save/_npy/keras49_9_test_x.npy',arr=x_test)
np.save('D:/study_data/_save/_npy/keras49_9_test_y.npy',arr=y_test)







