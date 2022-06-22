import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# [과제] 넘파이 리스트의 슬라이싱!! 7:3으로 잘라라 (6/21 과제)
x_train = x[:7] # [0:7] 과 같고 7번까지가 아닌 7번째 인수 전까지만 인용. 개념은 [시작:n+1]
x_test = x[-3:] # 시작과 끝은 명시하지 않아도 된다.
y_train = y[:7]
y_test =y[-3:]
#전체 중 70%만 트레인으로 활용 시 남은 30%는 버리는 데이터가 된다.
#그래서 랜덤으로 데이터를 골라 활용한다.
#x_train = np.array([1,2,3,4,5,6,7])
#x_test = np.array([8,9,10])
#y_train = np.array([1,2,3,4,5,6,7])
#y_test = np.array([8,9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(3))
model.add(Dense(1))


#3, 컴파일, 훈련

model.compile(loss='mse', optimizer="adam")
model.fit(x_train, y_train, epochs=100,batch_size=1)

#4. 평가,예측

loss = model.evaluate(x_test, y_test)
print('loss :', loss)
result = model.predict([11])
print('[11]의 예측값 :',result)

#loss : 4.2443084637133754e-12
#[11]의 예측값 : [[11.000002]]


