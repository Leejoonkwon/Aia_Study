import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25,shuffle=True ,random_state=100)
print(y_train,y_test)


#2. 모델 구성
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron ,LogisticRegression 
#LogisticRegression은 유일하게 Regression이름이지만 분류 모델이다.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier #공부하자 
from sklearn.ensemble import RandomForestClassifier #공부하자 

# model = LinearSVC() 
# model = LogisticRegression() 
# model = KNeighborsClassifier() 
# model = DecisionTreeClassifier() 
model = RandomForestClassifier() 

#3. 컴파일,훈련
model.fit(x_train,y_train) 


#4.  평가,예측
results = model.score(x_test,y_test) #분류 모델과 회귀 모델에서 score를 쓰면 알아서 값이 나온다 
print("results :",results)
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
