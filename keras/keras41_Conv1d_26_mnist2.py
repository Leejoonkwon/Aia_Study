from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout,LSTM,Conv1D
from sklearn.model_selection import train_test_split
from keras.datasets import mnist,cifar10
import pandas as pd
import numpy as np


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
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(60000, 28,28)
x_test = x_test.reshape(10000, 28,28)
print(np.unique(y_train,return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
#  array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#       dtype=int64))

y_train = pd.get_dummies((y_train)) 
y_test = pd.get_dummies((y_test))



#2. 모델 구성
model = Sequential()
# model.add(LSTM(10,input_shape=(28,28)))
model.add(Conv1D(10,2,input_shape=(28,28)))
model.add(Flatten())
model.add(Dense(100,activation='swish'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='swish'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))
model.summary()
import datetime
date = datetime.datetime.now()
print(date)

date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)
import time
start_time = time.time()

#3. 컴파일 훈련
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
filepath = './_ModelCheckPoint/K24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
earlyStopping = EarlyStopping(monitor='loss', patience=5, mode='min', 
                              verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
                      save_best_only=True, 
                      filepath="".join([filepath,'k25_', date, '_mnist2_', filename])
                    )
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=3500, 
                validation_split=0.33,
                callbacks = [earlyStopping,mcp],
                verbose=2
                )

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score,accuracy_score
# print(y_test.shape)

# y_test = np.argmax(y_test,axis=1)
print(y_predict)

y_predict = np.argmax(y_predict,axis=1)
# y_test와 y_predict의  shape가 일치해야한다.
print(y_test) #[10000 rows x 10 columns]
y_predict = pd.get_dummies((y_predict))

acc = accuracy_score(y_test, y_predict)
print('acc 스코어 :', acc)
end_time =time.time()-start_time
print("걸린 시간 :",end_time)
# loss : [0.03594522178173065, 0.9915000200271606]
# acc 스코어 : 0.9915

# LSTM
# loss : [0.3141157031059265, 0.8978999853134155]
# acc 스코어 : 0.8979

# Conv1d
# acc 스코어 : 0.9793
# 걸린 시간 : 19.142319917678833

