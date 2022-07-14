from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D ,Dropout,LSTM,Conv1D
from keras.datasets import mnist,cifar10,cifar100,fashion_mnist
import pandas as pd
import numpy as np


#1. 데이터 전처리

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape,y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape,y_test.shape) #(10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28* 28* 1)
x_test = x_test.reshape(10000, 28* 28* 1)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = StandardScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
#이전에 스케일링에서 dimension 오류가 뜨던 이유는 스케일링 작업은 1차원 타입의 데이터만 수정할 수 있기 때문이다.
#해결법은 스케일링 하기전 reshape로 (N,X)의 shape로 만들어 스케일링 후 다시 reshape를 통해 
#input_shape로 쓸 수 있게 만들면 가능하며 성능도 당연히 좋아진다.
print(x_train.shape) #(60000, 28, 28, 1)
print(x_test.shape) #(10000, 28, 28, 1)
print(np.unique(y_train,return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
#        dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000],
#       dtype=int64))
from tensorflow.keras.utils import to_categorical 
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test) 

# y_train = pd.get_dummies((y_train)) 
# y_test = pd.get_dummies((y_test))

print(y_train.shape) #(50000, 10)
print(y_test.shape) #(10000, 10)


#2. 모델 구성
model = Sequential()
# model.add(LSTM(10,input_shape=(28,28)))
model.add(Conv1D(10,2,input_shape=(28,28)))
model.add(Flatten())
model.add(Dense(100,activation='swish'))
model.add(Dropout(0.3))
model.add(Dense(100,activation='swish'))
model.add(Dropout(0.3))
model.add(Dense(100,activation='swish'))
model.add(Dropout(0.3))
model.add(Dense(100,activation='swish'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))
model.summary()
filepath = './_ModelCheckPoint/K24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
import time
start_time = time.time()

#3. 컴파일 훈련
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
earlyStopping = EarlyStopping(monitor='loss', patience=50, mode='min', 
                              verbose=1,restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(x_train, y_train, epochs=50, batch_size=2500, 
                callbacks = [earlyStopping],
                validation_split=0.25,
                verbose=2
                )

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score,accuracy_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)
y_test = np.argmax(y_test,axis=1)
y_predict = np.argmax(y_predict,axis=1)
# y_test와 y_predict의  shape가 일치해야한다.

acc = accuracy_score(y_test, y_predict)
print('acc 스코어 :', acc)
end_time =time.time()-start_time
print("걸린 시간 :",end_time)
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.rcParams['font.family']='Malgun Gothic'
# matplotlib.rcParams['axes.unicode_minus']=False
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'],marker='.',c='red',label='loss') #순차적으로 출력이므로  y값 지정 필요 x
# plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss')
# plt.grid()
# plt.title('loss와 val_loss') #맥플러립 한글 깨짐 현상 알아서 해결해라 
# plt.ylabel('loss')
# plt.xlabel('epochs')
# # plt.legend(loc='upper right')
# plt.legend()
# plt.show()

# loss : [0.7810052037239075, 0.89410001039505] 
# r2스코어 : 0.7887108142215705
# acc 스코어 : 0.8941
 
# loss : [0.28547945618629456, 0.9129999876022339]
# r2스코어 : 0.8546087436662431
# acc 스코어 : 0.913

# LSTM
# loss : [4.3096160888671875, 0.03200000151991844]
# acc 스코어 : 0.032

# Conv1d




