import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
import tensorflow as tf
tf.random.set_seed(66)
from sklearn.svm import LinearSVC #서포트벡터머신 알아서 공부해라 !!!
#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25,shuffle=True ,random_state=100)
print(y_train,y_test)


#2. 모델 구성
model = LinearSVC() # DL과 다르게 단층 레이어  구성으로 연산에 걸리는 시간을 비교할 수 없다.


#3. 컴파일,훈련
model.fit(x_train,y_train) 


#4.  평가,예측

results = model.score(x_test,y_test) #분류 모델과 회귀 모델에서 score를 쓰면 알아서 값이 나온다 
#ex)분류는 ACC 회귀는 R2스코어
print("results :",results)
y_predict = model.predict(x_test)
# y_test = np.argmax(y_test,axis=1)
print(y_predict)

# y_predict = np.argmax(y_predict,axis=1)
# # y_test와 y_predict의  shape가 일치해야한다.
# print(y_predict)


acc = accuracy_score(y_test, y_predict)
print('acc 스코어 :', acc)
#acc 스코어 : 0.9736842105263158

