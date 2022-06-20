import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x = np.array([1,2,3,5,4])
y = np.array([1,2,3,4,5])

# [실습] 맹그러봐! [6]을 예측한다.

#2. 모델
model = Sequential()
model.add(Dense(21,input_dim=1))
model.add(Dense(12))
model.add(Dense(11))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(8))
model.add(Dense(8))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae',optimizer='adam')
model.fit(x, y, epochs=2225)

#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss :', loss)

result = model.predict([6])
print('6의 예측값 :',result)

#loss : 0.4170146584510803
#6의 예측값 : [[6.0140305]]