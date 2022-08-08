import numpy as np  
from sklearn.datasets import load_iris    


#1. 데이터
datasets = load_iris()
x =  datasets.data 
y =  datasets.target   

from sklearn.model_selection import train_test_split 

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.8,shuffle=True,random_state=123)


#2. 모델 구성
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier 
from xgboost import XGBClassifier 

# model = DecisionTreeClassifier() 
# model = RandomForestClassifier() 
# model  = GradientBoostingClassifier()
model  = XGBClassifier()
#3. 훈련 
model.fit(x_train,y_train)

#4. 평가,예측

result = model.score(x_test,y_test)
print("model.score :",result)

from sklearn.metrics import accuracy_score, r2_score 
y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)
print('accuracy :',acc)


print("=============================")
print(model,':',model.feature_importances_)

# DecisionTreeClassifier()      : [0.01253395 0.01253395 0.5618817  0.4130504 ]
# RandomForestClassifier()      : [0.10164772 0.02537274 0.41392745 0.45905209]
# GradientBoostingClassifier()  : [0.000993   0.02414597 0.6304839  0.34437714]
# XGBClassifier()               : [0.0089478  0.01652037 0.75273126 0.22180054]
# feature_importtances는 y값에 영향을 미치는 정도를 수치화한다.
# iris는 컬럼이 4개지만 데이터가 더 방대해질경우 필요한 것과 불필요한 것을 분리할 수 있다.
# 여러가지 모델을 비교해서 영향을 확인해야한다.

