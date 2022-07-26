import numpy as np      
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn import datasets

#1. 데이터

train_datagen = ImageDataGenerator(
    rescale=1./255,               # MinMax 스케일링과 같은 개념 
    horizontal_flip=True,         # 인풋을 무작위로 가로로 뒤집습니다.
    vertical_flip=True,           # 인풋을 무작위로 세로로 뒤집습니다.
    width_shift_range=0.1,        #부동소수점: < 1인 경우 전체 가로넓이에서의 비율, >= 1인 경우 픽셀의 개수입니다.
    height_shift_range=0.1,       #부동소수점: < 1인 경우 전체 세로높이에서의 비율, >= 1인 경우 픽셀의 개수입니다.
    rotation_range=5,             #정수. 무작위 회전의 각도 범위입니다
    zoom_range=1.2,               #부동소수점 혹은 [하한, 상산]. 무작위 줌의 범위입니다(이미지를 확대 및 축소)
    shear_range=0.7,              #부동소수점. 층밀리기의 강도입니다. 이미지를 찌그러 트린다.(회전과 다름)
    fill_mode='nearest'          # {"constant", "nearest", "reflect" 혹은 "wrap"} 중 하나. 
  )                               # 회전 축소 등으로 이미지에 여백이생겼을때 채우는 방법입니다.

test_datagen = ImageDataGenerator(
    rescale=1./255
) # test data는 원형을 건드리지 않는다.훈련한 데이터와 비교하기 위해서 변형 X
xy_train = train_datagen.flow_from_directory(
    'd:/study_data/_data/image/brain/train/',
    target_size=(150,150),
    batch_size=160,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,) # 경로 및 폴더 설정
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/image/brain/test/',
     target_size=(150,150), #target_size는 본인 자유 
    batch_size=160,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=False,
    ).next() # Found 120 images belonging to 2 classes.
print("==================")
print(xy_train[0][1], xy_train[0][1].shape) #(160, 150, 150, 1) (160,) 1027장??

# print(xy_test[0][0].shape, xy_test[0][1].shape) #(120, 150, 150, 1) (120,)
x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

augument_size = 40000
randidx = np.random.randint(x_train.shape[0],size=augument_size)

x_augumented = x_train[randidx].copy()
y_augumented = y_train[randidx].copy()


print(x_augumented.shape)  #(400, 28, 28)
print(y_augumented.shape) #(400,) 50000, 32, 32, 3)
print(x_train.shape) #(160, 150, 150, 1)




xy_df2 = train_datagen.flow(x_train,y_train,
                                  batch_size=augument_size,shuffle=False)
x_df = np.concatenate((x_train,x_augumented))
y_df = np.concatenate((y_train,y_augumented))
# print(x_df.shape) #(64000, 28, 28, 1)

xy_df3 = test_datagen.flow(x_df,y_df,
                       batch_size=augument_size,shuffle=False)
from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test =train_test_split(xy_df3[0][0],xy_df3[0][1],train_size=0.75,shuffle=False)
np.save('D:/study_data/_save/_npy/keras49_5_train_x.npy',arr=xy_df3[0][0])
np.save('D:/study_data/_save/_npy/keras49_5_train_y.npy',arr=xy_df3[0][1])
np.save('D:/study_data/_save/_npy/keras49_5_test_x.npy',arr=x_test)
np.save('D:/study_data/_save/_npy/keras49_5_test_y.npy',arr=y_test)


