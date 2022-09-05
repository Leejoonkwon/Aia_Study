# 맹그러봐
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout 
from sklearn.model_selection import train_test_split
from keras.datasets import mnist,cifar10
import pandas as pd
import numpy as np
import time
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import adam, adadelta, adagrad, adamax
from tensorflow.python.keras.optimizer_v2 import rmsprop,nadam
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.keras.utils import to_categorical
#1. 데이터 전처리
# datasets = mnist.load_data()
# x = datasets.data #데이터를 리스트 형태로 불러올 때 함
# y = datasets.target
# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25,shuffle=True ,random_state=100)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape,y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape,y_test.shape) #(60000,) (10000,)

# x_train = x_train.reshape(60000, 28*28*1)
# x_test = x_test.reshape(10000, 28*28*1)
x_train = x_train.reshape(60000, 28*28*1)
x_test = x_test.reshape(10000, 28*28*1)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(60000, 28,28,1)
x_test = x_test.reshape(10000, 28,28,1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3),   # 출력(4,4,10)                                    
                 padding='same',
                 input_shape=(28, 28, 1)))    #(batch_size, row, column, channels)     
model.add(MaxPooling2D())
model.add(Conv2D(32, (2,2),  #인풋쉐이프에 행값은 디폴트는 32
                 padding = 'same',         # 디폴트값(안준것과 같다.) 
                 activation= 'swish'))    # 출력(3,3,7)       
model.add(MaxPooling2D())
model.add(Conv2D(100, (2,2), 
                 padding = 'same',         # 디폴트값(안준것과 같다.) 
                 activation= 'swish'))    # 출력(3,3,7)      
model.add(Flatten()) # (N, 63) 위치와 순서는 바뀌지 않아야한다.transpose와 전혀 다르다.
model.add(Dense(100))
model.add(Dense(100,activation='swish'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='swish'))
model.add(Dropout(0.3))
model.add(Dense(10,activation='softmax'))

#3. 컴파일 훈련
optilist = [adam.Adam,adadelta.Adadelta,adagrad.Adagrad,
        adagrad.Adagrad,adamax.Adamax,rmsprop.RMSprop,nadam.Nadam]
# Defaults to 0.001.
start_time = time.time()
emptylist =[]
for i in optilist:
#     learning_rate = 0.01
#     optimizer = i(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=i(lr=0.01))
    model.fit(x_train,y_train,epochs=50,batch_size=7000,verbose=0)
    #4. 평가, 예측
    loss = model.evaluate(x_test,y_test)
    y_predict = model.predict(x_test)
    y_predict = np.argmax(y_predict,axis=1)
    y_predict = to_categorical(y_predict)
    print(y_predict)    
    print(i.__name__)
    print(accuracy_score(y_test,y_predict))

# loss : [0.03594522178173065, 0.9915000200271606]
# acc 스코어 : 0.9915
