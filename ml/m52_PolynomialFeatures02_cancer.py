import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split,KFold 
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.datasets import load_boston,load_breast_cancer
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.pipeline import make_pipeline

#1. 데이터
datasets = load_breast_cancer()
x, y =datasets.data, datasets.target 
print(x.shape,y.shape) #(506, 13) (506,)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8,random_state=1234,
)


#2. 모델
model = make_pipeline(StandardScaler(),
                      XGBClassifier())

#3. 훈련
model.fit(x_train,y_train)


#4. 평가,예측
kfold = KFold(n_splits=5,random_state=123,shuffle=True)
print('기냥 스코어 :',model.score(x_test,y_test))
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,x_train,y_train,cv=kfold,scoring='r2')
print("기냥 CV : ",scores)
print("기냥 CV 엔빵 : ",np.mean(scores))
model2 = make_pipeline


################## PolynomialFeatures 후 

pf = PolynomialFeatures(degree=2,include_bias=False)
xp = pf.fit_transform(x)
print(xp.shape) #(506, 105)

x_train,x_test,y_train,y_test = train_test_split(
    xp,y,train_size=0.8,random_state=1234,
)

#2. 모델
model = make_pipeline(StandardScaler(),
                      XGBClassifier())

#3. 훈련
model.fit(x_train,y_train)


#4. 평가,예측

print('poly 스코어 :',model.score(x_test,y_test))
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,x_train,y_train,cv=kfold,scoring='r2')
print("폴리 CV : ",scores)
print("폴리 CV 엔빵 : ",np.mean(scores))
# (569, 30) (569,)
# 기냥 스코어 : 0.9385964912280702
# 기냥 CV :  [0.77939394 0.91025641 0.81926514 0.85913313 1.        ]
# 기냥 CV 엔빵 :  0.8736097241154781
# (569, 495)
# poly 스코어 : 0.9385964912280702
# 폴리 CV :  [0.77939394 0.91025641 0.81926514 1.         0.90215054]
# 폴리 CV 엔빵 :  0.8822132062553628