import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.datasets import load_boston
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.pipeline import make_pipeline

#1. 데이터
datasets = load_boston()
x, y =datasets.data, datasets.target 
print(x.shape,y.shape) #(506, 13) (506,)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8,random_state=1234,
)


#2. 모델
model = make_pipeline(StandardScaler(),
                      CatBoostRegressor(verbose=False))
       
#3. 훈련
model.fit(x_train,y_train)
   

#4. 평가,예측

print('model.score :',model.score(x_test,y_test))

model2 = make_pipeline
# model.score : 0.9244748735827965





