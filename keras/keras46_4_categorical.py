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
    fill_mode='nearest' )         # {"constant", "nearest", "reflect" 혹은 "wrap"} 중 하나. 
                                  # 회전 축소 등으로 이미지에 여백이생겼을때 채우는 방법입니다.

test_datagen = ImageDataGenerator(
    rescale=1./255
) # test data는 원형을 건드리지 않는다.훈련한 데이터와 비교하기 위해서 변형 X
xy_train = train_datagen.flow_from_directory(
    'd:/_data/image/brain/train/',
    target_size=(200,200),
    batch_size=5,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=True,) # 경로 및 폴더 설정
# Found 160 images belonging to 2 classes.
xy_test = test_datagen.flow_from_directory(
    'd:/_data/image/brain/test/',
     target_size=(200,200), #target_size는 본인 자유 
    batch_size=5,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=True,
    ) # Found 120 images belonging to 2 classes.


# 현재 5,200,200,1의 데이터가 32덩어리

#2. 모델 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D,Flatten,Dense,MaxPool2D

model = Sequential()
model.add(Conv2D(32,(2,2),input_shape=(200,200,1),padding='same',activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(2,activation='softmax'))

#3. 컴파일,훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# model.fit(xy_train[0][0],xy_train[0][1]) 배치를 최대로 잡을 경우 가능한 구조
hist = model.fit_generator(xy_train,epochs=1000,
                    validation_data=xy_test,
                    steps_per_epoch=32,
                    validation_steps=4) # 배치가 최대 아닐 경우 사용

accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss =  hist.history['loss']
val_loss =  hist.history['val_loss']

print('loss :',loss[-1])
print('val_loss :',val_loss[-1])
print('accuracy :',accuracy[-1])
print('val_accuracy :',val_accuracy[-1])

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'],marker='.',c='red',label='loss') #순차적으로 출력이므로  y값 지정 필요 x
plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss')
plt.grid()
plt.title('show') #맥플러립 한글 깨짐 현상 알아서 해결해라 
plt.ylabel('loss')
plt.xlabel('epochs')
# plt.legend(loc='upper right')
plt.legend()
plt.show()
# loss : 0.32224059104919434
# val_loss : 1.612489104270935
# accuracy : 0.8687499761581421
# val_accuracy : 0.6000000238418579
############### categorical 로 바꿀 시
# loss : 0.259000688791275
# val_loss : 0.21025490760803223
# accuracy : 0.90625
# val_accuracy : 0.949999988079071


