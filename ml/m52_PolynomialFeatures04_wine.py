import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split,KFold 
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.datasets import load_boston,load_wine
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.pipeline import make_pipeline

#1. 데이터
datasets = load_wine()
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
# model.score : 0.8745129304823863

# (178, 13) (178,)
# 기냥 스코어 : 0.8888888888888888
# 기냥 CV :  [0.72222222 1.         1.         0.94646272 0.94852941]
# 기냥 CV 엔빵 :  0.9234428698184182
# (178, 104)
# poly 스코어 : 0.9166666666666666
# 폴리 CV :  [0.94444444 0.92487047 1.         1.         0.89705882]
# 폴리 CV 엔빵 :  0.95327474685902