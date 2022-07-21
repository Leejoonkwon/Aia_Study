from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout 
from keras.datasets import mnist,cifar10
import pandas as pd
import numpy as np


#1. 데이터 전처리

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape,y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape,y_test.shape) #(10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(50000, 32*32* 3)
x_test = x_test.reshape(10000, 32*32* 3)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape) #(50000, 32, 32, 3, 1)
print(x_test.shape) #(10000, 32, 32, 3, 1)
print(np.unique(y_train,return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
#        dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
#       dtype=int64))
print(y_train.shape) #(50000, 1)
print(y_test.shape) #(10000, 1)

from tensorflow.keras.utils import to_categorical 
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test) 

# y_train = pd.get_dummies((y_train)) 
# y_test = pd.get_dummies((y_test))
print(y_train.shape) #(50000, 10)
print(y_test.shape) #(10000, 10)


#2. 모델 구성
# model = Sequential()
# # model.add(Flatten()) #  해도 돌아감
# model.add(Dense(100,input_shape=(3072,),activation='swish'))
# model.add(Dropout(0.3))
# model.add(Dense(100,activation='swish'))
# model.add(Dropout(0.3))
# model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(10, activation='softmax'))
# model.summary()
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input
input1 = Input(shape=(3072,))
dense1 = Dense(100)(input1)
dense2 = Dense(100,activation='relu')(dense1)
dense3 = Dense(100,activation='relu')(dense2)
output1 = Dense(10,activation='softmax')(dense3)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일 훈련
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
earlyStopping = EarlyStopping(monitor='loss', patience=150, mode='min', 
                              verbose=1,restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=150, batch_size=4500, 
                callbacks = [earlyStopping],
                verbose=2
                )
x_train = np.load('D:/study_data/_save/_npy/keras49_2_train_x.npy')
y_train = np.load('D:/study_data/_save/_npy/keras49_2_train_y.npy')
x_test = np.load('D:/study_data/_save/_npy/keras49_2_test_x.npy')
y_test = np.load('D:/study_data/_save/_npy/keras49_2_test_y.npy')


#2. 모델 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D,Flatten,Dense,MaxPool2D

model = Sequential()
model.add(Conv2D(32,(2,2),input_shape=(28,28,1),padding='same',activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(10,activation='softmax'))

#3. 컴파일,훈련
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=4,verbose=2,validation_split=0.25,batch_size=500)
# hist = model.fit_generator(x_train,y_train,epochs=2,
#                     validation_split=0.25,
#                     steps_per_epoch=32,
#                     validation_steps=4) # 배치가 최대 아닐 경우 사용

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
print('y_predict :', y_predict)

#증폭 후 
# loss : [0.35870376229286194, 0.8970000147819519] 
