from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,9,9,12,14,6,8,12,14,17,16,12,15,20])

x_train, x_test ,y_train, y_test = train_test_split(x, y, train_size=0.7,shuffle=True,random_state=66)
#줄을 같은 라인에 사용시 독립된 코드로 인식 종속된 코드로 작성 시 줄을 다르게 작성해야 오류 발생 X
#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일,훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=350, batch_size=1)

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x)

import matplotlib.pyplot as plt

plt.scatter(x,y)
plt.plot(x, y_predict, color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

#tep - loss: 2.1752
#loss : 2.17524790763855

