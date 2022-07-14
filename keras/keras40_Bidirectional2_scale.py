import numpy as np
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,LSTM,GRU



#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]
              ,[5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


#1. 데이터

print(x.shape,y.shape) #(7, 3) (7,)
print(y) #[4 5 6]
#RNN 인풋 쉐이프 (행,열,반복할 열을 자르는 단위 또는 몇개씩 자르는지) -> (N,3,1)

x = x.reshape(13,3,1)
print(x.shape,y.shape) #(7, 3, 1) (7,)

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units=100, input_shape=(3,1),activation='swish'))
model.add(Bidirectional(LSTM(60,return_sequences=True),input_shape=(3,1)))

model.add(LSTM(640,return_sequences=True))
model.add(LSTM(640,return_sequences=True))
model.add(GRU(640,activation='relu'))
model.add(Dense(16,activation='relu')) 
model.add(Dense(1,activation='relu'))
#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x, y, epochs=200)


#4. 평가, 예측
loss = model.evaluate(x,y)
y_predict = np.array([50,60,70]).reshape(1,3,1)
print(y_predict)
result = model.predict(y_predict)
print('loss ;',loss)
print('[50,60,70]의 결과 :',result)

# loss ; 5.540014535654336e-05
# [50,60,70]의 결과 : [[75.99887]]
