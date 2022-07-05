from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

 #1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))
#[실습]
x_train = x[:10]
y_train = y[:10]
x_test = x[10:12] # 평가 및 예측용 데이터
y_test = y[10:12]
x_val = x[12:] # 트레인 과정에서 검증 데이터
y_val = y[12:]
# x_train = np.array(range(1,11)) # 훈련용 데이터
# y_train = np.array(range(1,11))
# x_test = np.array([11,12,13]) # 평가 및 예측용 데이터
# y_test = np.array([11,12,13])
# x_val = np.array([14,15,16]) # 트레인 과정에서 검증 데이터
# y_val = np.array([14,15,16])

#2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1,
          validation_data=(x_val, y_val))
#검증 손실(val_loss)는 통상적으로 일반 손실(loss)보다 크게 예측되는 것이 정상이다.
#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss :',loss)

result = model.predict([17])
print("17의 예측값 :",result)