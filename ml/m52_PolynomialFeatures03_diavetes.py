import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split,KFold 
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.datasets import load_boston,load_diabetes
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.pipeline import make_pipeline

#1. 데이터
datasets = load_diabetes()
x, y =datasets.data, datasets.target 
print(x.shape,y.shape) #(506, 13) (506,)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8,random_state=1234,
)


#2. 모델
model = make_pipeline(StandardScaler(),
                      LinearRegression())

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
                      LinearRegression())

#3. 훈련
model.fit(x_train,y_train)


#4. 평가,예측

print('poly 스코어 :',model.score(x_test,y_test))
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,x_train,y_train,cv=kfold,scoring='r2')
print("폴리 CV : ",scores)
print("폴리 CV 엔빵 : ",np.mean(scores))
# model.score : 0.8745129304823863
# (442, 10) (442,)
# 기냥 스코어 : 0.46263830098374903
# 기냥 CV :  [0.51583902 0.46958492 0.38394311 0.54405474 0.56107605]
# 기냥 CV 엔빵 :  0.4948995654862613
# (442, 65)
# poly 스코어 : 0.384692640515067
# 폴리 CV :  [0.2773713  0.34022842 0.1802958  0.35399376 0.50029688]
# 폴리 CV 엔빵 :  0.33043722987647406
