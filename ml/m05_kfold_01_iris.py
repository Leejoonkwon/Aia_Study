import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets['target']

# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25,shuffle=True ,random_state=100)
# print(y_train,y_test)
# from sklearn.model_selection import KFold,cross_val_score

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)
#기존 train_test_split에서 데이터가 각자에 편향되는 문제점을 개선한다.
#전체 데이터를  N등분 후 N회에 걸쳐 교차 검증한다.그것이 kFold이다.
#cross_val_socre를 통해 N회 당 발생하는 validation score를 보여준다.
#교차 검증 스코어에서 일부가 잘 나왔다고 거기에 집중하지 않고 평균으로
#스코어를 출력하는 것이 좋다.과적합 방지 
#2. 모델 구성
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron ,LogisticRegression 
#LogisticRegression은 유일하게 Regression이름이지만 분류 모델이다.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier #공부하자 
from sklearn.ensemble import RandomForestClassifier #공부하자 

model = SVC() 

#3,4. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold)
# scores = cross_val_score(model, x, y, cv=5) # kfold의 튜닝없이 디폴트로 쓸 수 있음 

print('ACC :' ,scores,'\n cross_val_score : ',round(np.mean(scores),4))
# ACC : [0.96666667 0.96666667 1.         0.93333333 0.96666667]
#  cross_val_score :  0.9667


# model.fit(x_train,y_train) 


# #4.  평가,예측
# results = model.score(x_test,y_test) #분류 모델과 회귀 모델에서 score를 쓰면 알아서 값이 나온다 
# print("results :",results)
# y_predict = model.predict(x_test)
# print(y_predict)

# acc = accuracy_score(y_test, y_predict)
# print('acc 스코어 :', acc)

# model = LinearSVC() 
#acc 스코어 : 0.9736842105263158

# model = LogisticRegression() 
# results : 0.9473684210526315

# model = KNeighborsClassifier() 
# results : 0.9736842105263158

# model = DecisionTreeClassifier() 
# results : 0.9473684210526315

# model = RandomForestClassifier() 
# results : 0.9473684210526315
