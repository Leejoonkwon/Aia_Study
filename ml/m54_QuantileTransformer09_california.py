import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split,KFold 
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import RobustScaler,QuantileTransformer # 이상치에 강함
from sklearn.preprocessing import PowerTransformer
from sklearn.datasets import load_boston,fetch_california_housing
from sklearn.metrics import r2_score,accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from catboost import CatBoostRegressor
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
#1. 데이터
datasets = fetch_california_housing()
x, y =datasets.data, datasets.target 
print(x.shape,y.shape) #(506, 13) (506,)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8,random_state=1234,
)

aaa =[StandardScaler(),MinMaxScaler(),RobustScaler(),
      QuantileTransformer(),PowerTransformer(method='yeo-johnson'),
    #   PowerTransformer(method='box-cox')
      ]
for i in aaa:
    scaler = i
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = RandomForestRegressor()
    model.fit(x_train,y_train)
    y_predict =model.predict(x_test)
    results = r2_score(y_test,y_predict)
    print("{0} : {1:4f} ".format(i,round(results, 4)))

# StandardScaler() : 0.805900 
# MinMaxScaler() : 0.807500 
# RobustScaler() : 0.802200 
# QuantileTransformer() : 0.804700 
# PowerTransformer() : 0.803000 

# Catboost
# StandardScaler() : 0.845600 
# MinMaxScaler() : 0.845600 
# RobustScaler() : 0.845600 
# QuantileTransformer() : 0.845600 
# PowerTransformer() : 0.845600
