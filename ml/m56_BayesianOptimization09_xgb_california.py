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
    x,y, train_size=0.8,random_state=123
)
from xgboost import XGBClassifier,XGBRegressor
from bayes_opt import BayesianOptimization

xgb_parameters = {'max_depth': (3, 6),
              'min_child_weight' : (1, 50),
              'subsample' : (0.5, 1),
              'colsample_bytree' : (0.5, 1),
              'reg_alpha' : (0.01, 50),
              'reg_lambda' : (0.001, 10)
              } # 디폴트 6 

def xgb_hamsu(max_depth,min_child_weight,
              subsample,colsample_bytree,reg_lambda,reg_alpha,) :
    params = {
        'n_estimators':500,"learning_rate":0.01,
        'max_depth' : int(round(max_depth)),  #무조건 정수
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample,1),0), # 0~1 사이의 값이 들어가도록 한다.
        'colsample_bytree' : max(min(colsample_bytree,1),0),
        'reg_lambda' : max(reg_lambda, 0), # 무조건 양수만 
        'reg_alpha' : max(reg_alpha, 0),
          }
    
    # *여러개의 인자를 받겠다.
    # **키워드 받겠다(딕셔너리형태)
    model = XGBRegressor(**params)
    
    model.fit(x_train,y_train,
              # eval_set=[(x_train,y_train),(x_test,y_test)],
              verbose=0,
              # eval_metric='mlogloss',
              # early_stopping_rounds=50,
              )
    y_predict = model.predict(x_test)
    results = r2_score(y_test,y_predict)
    
    return results
xgb_bo = BayesianOptimization(f=xgb_hamsu,
                              pbounds=xgb_parameters,
                              random_state=123)

xgb_bo.maximize(init_points=2,
                n_iter=30)

print(xgb_bo.max)
    
# StandardScaler() : 0.805900 
# MinMaxScaler() : 0.807500 
# RobustScaler() : 0.802200 
# QuantileTransformer() : 0.804700 
# PowerTransformer() : 0.8030
# Catboost     
# StandardScaler() : 0.8456
# MinMaxScaler() : 0.845600    
# RobustScaler() : 0.845600    
# QuantileTransformer() : 0.845600 
# PowerTransformer() : 0.845600
