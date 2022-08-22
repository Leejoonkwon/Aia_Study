import numpy as np 
import pandas as pd 
from sklearn.datasets import load_breast_cancer 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = load_breast_cancer()
x,y = datasets.data, datasets.target # 이렇게도 된다.초등생용 문법
print(x.shape,y.shape) #(569, 30) (569,)

x_train,x_test,y_train,y_test =train_test_split(x,y,train_size=0.8,
        random_state=123,shuffle=True,stratify=y)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
model = BaggingClassifier(DecisionTreeClassifier(),
                          n_estimators=100,#해당 모델을 100번 훈련한다.
                          n_jobs=-1,
                          random_state=123
                          )
# Bagging 할 때는 스케일링이 무조건 필요하다.
# Bagging(Bootstrap Aggregating)
# 한가지 모델을 여러번 훈련한다.대표적인 Ensemble 모데 랜덤포레스트

#3. 훈련
model.fit(x_train,y_train)


#4. 평가, 예측
print('model.score :',model.score(x_test,y_test))
######Bagging 후 ACC
# model.score : 0.9736842105263158




