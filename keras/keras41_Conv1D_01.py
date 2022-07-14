import numpy as np
import time
from tensorflow.python.keras.models import Sequential
# from tensorflow.keras.layers import Bidirectional
from tensorflow.python.keras.layers import Dense,LSTM,Conv1D,Flatten #Conv1d는 3차원!
# 레이어 위치 바꿔서도 해보라~~~

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
#y = ??

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]])
y = np.array([4,5,6,7,8,9,10])

print(x.shape,y.shape) #(7, 3) (7,)
print(y) #[4 5 6]
#RNN 인풋 쉐이프 (행,열,반복할 열을 자르는 단위 또는 몇개씩 자르는지) -> (N,3,1)

x = x.reshape(7,3,1)
print(x.shape,y.shape) #(7, 3, 1) (7,)
start_time = time.time()
#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units=100, input_shape=(3,1),activation='swish'))
# model.add(LSTM(units=10, return_sequences=False,input_shape=(3,1),activation='swish'))
model.add(Conv1D(10,2,input_shape=(3,1)))
model.add(Flatten())
model.add(Dense(3,activation='relu')) 
model.add(Dense(1,activation='relu'))
model.summary()

# LSTM Total params: 517
# Conv1d Total params: 97

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x, y, epochs=50)

#4. 평가, 예측
loss = model.evaluate(x,y)
y_predict = np.array([8,9,10]).reshape(1,3,1) 
print(y_predict)
# [[[ 8]
#   [ 9]
#   [10]]]
result = model.predict(y_predict)
print('loss ;',loss)
print('[8,9,10]의 결과 :',result)

# loss ; 0.002228141063824296
# [8,9,10]의 결과 : [[10.9396715]]







