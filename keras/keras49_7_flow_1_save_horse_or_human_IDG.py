import numpy as np      
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn import datasets

#1. 데이터

train = ImageDataGenerator(
    rescale=1./255,               # MinMax 스케일링과 같은 개념 
    # horizontal_flip=True,         # 인풋을 무작위로 가로로 뒤집습니다.
    # vertical_flip=True,           # 인풋을 무작위로 세로로 뒤집습니다.
    # width_shift_range=0.1,        #부동소수점: < 1인 경우 전체 가로넓이에서의 비율, >= 1인 경우 픽셀의 개수입니다.
    # height_shift_range=0.1,       #부동소수점: < 1인 경우 전체 세로높이에서의 비율, >= 1인 경우 픽셀의 개수입니다.
    # rotation_range=5,             #정수. 무작위 회전의 각도 범위입니다
    # zoom_range=1.2,               #부동소수점 혹은 [하한, 상산]. 무작위 줌의 범위입니다(이미지를 확대 및 축소)
    # shear_range=0.7,              #부동소수점. 층밀리기의 강도입니다. 이미지를 찌그러 트린다.(회전과 다름)
    # fill_mode='nearest'          # {"constant", "nearest", "reflect" 혹은 "wrap"} 중 하나. 
  )                               # 회전 축소 등으로 이미지에 여백이생겼을때 채우는 방법입니다.

test_datagen = ImageDataGenerator(
    rescale=1./255)
# ) # test data는 원형을 건드리지 않는다.훈련한 데이터와 비교하기 위해서 변형 X
xydata = train.flow_from_directory(
    'D:\study_data\_data\image\horse-or-human\horse-or-human',
    target_size=(150,150),
    class_mode='binary',
    batch_size=20000,
    shuffle=True,) # 경로 및 폴더 설정

# print(xydata[0][0],xydata[0][0].shape) # (32, 150, 150, 3)
print(xydata[0][0].shape) # (1027, 150, 150, 3)
print(xydata[0][1].shape) # (1027,)
x_train = xydata[0][0][0:730]
y_train = xydata[0][1][0:730]
x_test = xydata[0][0][730:]
y_test = xydata[0][1][730:]

print(x_train.shape) # (730, 150, 150, 3)
print(y_train.shape) # (730,)


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
np.save('D:/study_data/_save/_npy/keras49_7_train_x.npy',arr=xy_df3[0][0])
np.save('D:/study_data/_save/_npy/keras49_7_train_y.npy',arr=xy_df3[0][1])
np.save('D:/study_data/_save/_npy/keras49_7_test_x.npy',arr=x_test)
np.save('D:/study_data/_save/_npy/keras49_7_test_y.npy',arr=y_test)



