from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,14,13,15,16,17,18,20,19])

x_train, x_test ,y_train, y_test = train_test_split(x, y, train_size=0.7,shuffle=True,random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(24, input_dim=1))
model.add(Dense(12))
model.add(Dense(9))
model.add(Dense(6))
model.add(Dense(6))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일,훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=1)

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)
print('r2스코어 :', r2)

# tep - loss: 0.0063
# loss : 0.006292529869824648
# r2스코어 : 0.9936993631962853



# import matplotlib.pyplot as plt

# plt.scatter(x,y)
# plt.plot(x, y_predict, color='red')
# plt.show()







