# xor gate는 참일 때 0을 반환 거짓일 때 1 반환
import numpy as np 
from sklearn.svm    import LinearSVC,SVC
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import Perceptron

#1. 데이터
x_data  = [[0,0],[0,1],[1,0],[1,1]] #(4,2) # AND게이트는 논리곱을 구현하는 기본 디지털 논리 게이트
y_data  = [0, 1, 1, 0] #(4,)

#2. 모델 
# model = LinearSVC() 
# model = Perceptron()
model = SVC()

#3. 훈련
model.fit(x_data,y_data)

#4. 평가,예측

y_predict = model.predict(x_data)
print(x_data,"의 예측결과",y_predict)

results = model.score(x_data,y_data)
print("model.score :",results)

acc = accuracy_score(y_data,y_predict)
print("acc score :",acc)

# model.score : 1.0
# acc score : 1.0
