import numpy as np 
import pandas as pd 
from sklearn.datasets import load_breast_cancer,load_iris
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = load_wine()
x,y = datasets.data, datasets.target # 이렇게도 된다.초등생용 문법
print(x.shape,y.shape) #(178, 13) (178,)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier,RandomForestClassifier

from sklearn.preprocessing import StandardScaler
x_train,x_test,y_train,y_test = train_test_split(
    datasets.data,datasets.target,train_size=0.8,shuffle=True,random_state=123,
    stratify=datasets.target)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델

xg = XGBClassifier(random_state=123)
lg = LGBMClassifier(random_state=123,
                    learning_rate=0.2)
cat = CatBoostClassifier(verbose=False,random_state=123)
rf =RandomForestClassifier()
model = VotingClassifier(
    estimators=[('XG', xg), ('LG', lg),('CAT', cat),
                ('RF',rf)],
    voting='soft'    # hard 옵션도 있다.
)

# hard 는 결과 A 0 B 0 C 1이라면 결과는 0으로 다수결에 따른다.
# soft 는 클래스파이어간의 평균으로 결정

#3. 훈련
model.fit(x_train,y_train)

#4. 평가,예측
y_predict = model.predict(x_test)

score = accuracy_score(y_test,y_predict)
print("보팅 결과 :",round(score,4))


classifiers = [cat,xg,lg,rf]
for model2 in classifiers:
    model2.fit(x_train,y_train)
    y_predict = model2.predict(x_test)
    score2 = accuracy_score(y_test,y_predict)
    class_name = model2.__class__.__name__ # 해당 커맨드로 이름 반환
    print('{0} 정확도 : {1:4f}'.format(class_name,score2))

######Bagging 후 acc Df
# model.score : 0.9722222222222222

# 보팅 결과 : 0.9722
# CatBoostClassifier 정확도 : 0.972222
# XGBClassifier 정확도 : 0.916667
# LGBMClassifier 정확도 : 0.972222
# RandomForestClassifier 정확도 : 0.972222





