import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split,KFold 
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.datasets import load_boston,load_iris
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor,CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline

#1. 데이터
datasets = load_iris()
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

# LinearRegression
# model.score : 0.7665382927362872
# CatBoostRegressor
# model.score : 0.9244748735827965

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
# (150, 4) (150,)
# 기냥 스코어 : 1.0
# 기냥 CV :  [0.93314763 0.91176471 0.89450549 0.89450549 0.94444444]
# 기냥 CV 엔빵 :  0.9156735543299528
# (150, 14)
# poly 스코어 : 1.0
# 폴리 CV :  [0.93314763 0.91176471 0.89450549 0.78901099 0.94444444]
# 폴리 CV 엔빵 :  0.8945746532310517

