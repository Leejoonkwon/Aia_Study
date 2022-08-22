import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split,KFold 
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import RobustScaler,QuantileTransformer # 이상치에 강함
from sklearn.preprocessing import PowerTransformer
from sklearn.datasets import load_boston,load_iris
from sklearn.metrics import r2_score,accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
#1. 데이터
datasets = load_iris()
x, y =datasets.data, datasets.target 
print(x.shape,y.shape) #(506, 13) (506,)


x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8,random_state=1234,
)

aaa =[StandardScaler(),MinMaxScaler(),RobustScaler(),
      QuantileTransformer(),PowerTransformer(method='yeo-johnson')]
for i in aaa:
    scaler = i
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = DecisionTreeClassifier()
    model.fit(x_train,y_train)
    y_predict =model.predict(x_test)
    results = r2_score(y_test,y_predict)
    print("기냥 결과 :",round(results, 4))

# scaler = PowerTransformer(method='yeo-johnson') # 디폴트
# scaler = PowerTransformer(method='box-cox')

# (150, 4) (150,)
# 기냥 결과 : 1.0
# 기냥 결과 : 1.0
# 기냥 결과 : 1.0
# 기냥 결과 : 1.0
# 기냥 결과 : 1.0