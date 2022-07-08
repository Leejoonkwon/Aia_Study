from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D #Flatten평평하게해라.  # 이미지 작업 conv2D 
from keras.datasets import mnist
import pandas as pd
import numpy as np


#1. 데이터 전처리

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape,y_train.shape) #(60000, 28, 28) (60000,)
print(y_train.shape,y_test.shape) #(60000,) (10000,)

x_train = x_train.reshape(60000, 28*28*1)
x_test = x_test.reshape(10000, 28*28*1)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = StandardScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(60000, 28,28,1)
x_test = x_test.reshape(10000, 28,28,1)
print(np.unique(y_train,return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
#  array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#       dtype=int64))

y_train = pd.get_dummies((y_train)) 
y_test = pd.get_dummies((y_test))



#2. 모델 구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3),   # 출력(4,4,10)                                       # 자르는 사이즈 (행,렬 규격.) 10= 다음레이어에 주는 데이터
                 padding='same',
                 input_shape=(28, 28, 1)))    #(batch_size, row, column, channels)       # N(장수) 이미지 5,5 짜리 1 흑백 3 칼라 
                                                                                           # kernel_size(2*2) * 바이어스(3) + 10(output)
model.add(MaxPooling2D())

 #    (kernel_size * channls) * filters = summary Param 개수(CNN모델)  
model.add(Conv2D(32, (2,2), 
                 padding = 'valid',         # 디폴트값(안준것과 같다.) 
                 activation= 'relu'))    # 출력(3,3,7)                                                     
model.add(Flatten()) # (N, 63)
model.add(Dense(32, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(10, activation= 'softmax'))
model.summary()

#3. 컴파일 훈련
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
earlyStopping = EarlyStopping(monitor='loss', patience=100, mode='min', 
                              verbose=1,restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1000, batch_size=10000, 
                validation_split=0.3,
                callbacks = [earlyStopping],
                verbose=2
                )

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)

# loss : [0.23423144221305847, 0.978600025177002]
# r2스코어 : 0.9565073802008651
