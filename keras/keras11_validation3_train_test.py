from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

 #1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))
#[실습]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.1875, shuffle = True, random_state = 68
 )

print(x_train)

x_train, x_val, y_train, y_val= train_test_split(
    x, y, train_size= 0.5, shuffle = True, random_state = 68
 )
print(x_train,x_test,x_val)
'''
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
'''

