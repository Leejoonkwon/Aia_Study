import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,SimpleRNN,Dropout,LSTM#이름부터 간단?

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

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(100, input_shape=(3,1)))
model.add(LSTM(units=200, input_length=3,input_dim=1,activation='relu')) #위와 같은 개념
# 4*units*(dim+bias+units)
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm (LSTM)                  (None, 10)                480
# model.add(SimpleRNN(10))
#ValueError :expected ndim=3, found ndim=2 3차원으로 이어져야 하지만 2차원으로 나와서 진행 불가 RNN은 연속레이어 사용 불가 
# [SimpleRNN] units : 10 -> 10* (1 + 1 +10) = 120
# [LSTM] units : 10 -> 4*10* (1 + 1 +10) = 120
# 4를 곱하는 이유는 연산이 4번 되어지기 때문이다. Cell State,input gate,output gate, forget gate
model.add(Dense(100,activation='swish')) 
model.add(Dense(100,activation='swish')) 
model.add(Dense(1,activation='swish'))
model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x, y, epochs=150)

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

# loss ; 0.0017791687278077006
# [8,9,10]의 결과 : [[11.057822]]








