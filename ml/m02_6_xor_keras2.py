# xor gate는 참일 때 0을 반환 거짓일 때 1 반환
import numpy as np 
from sklearn.svm    import LinearSVC
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import Perceptron
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense
#1. 데이터
x_data  = [[0,0],[0,1],[1,0],[1,1]] #(4,2) # AND게이트는 논리곱을 구현하는 기본 디지털 논리 게이트
y_data  = [0, 1, 1, 0] #(4,)

#2. 모델 
# model = LinearSVC() 
# model = Perceptron()
model = Sequential()
model.add(Dense(10,input_dim=2,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
#3. 컴파일, 훈련

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_data,y_data,epochs=100,batch_size=1)

#4. 평가,예측
results = model.evaluate(x_data,y_data)
print("metrics.score :",results)
y_predict = model.predict(x_data)
print(x_data,"의 예측결과",y_predict[0])

# metrics.score : [0.1372033804655075, 1.0]
# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과 [0.06781906]
