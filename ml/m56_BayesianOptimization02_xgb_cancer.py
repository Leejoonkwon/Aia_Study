import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split,KFold 
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import RobustScaler,QuantileTransformer # 이상치에 강함
from sklearn.preprocessing import PowerTransformer
from sklearn.datasets import load_boston,load_breast_cancer
from sklearn.metrics import r2_score,accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from catboost import CatBoostRegressor
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
#1. 데이터
datasets = load_breast_cancer()
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
    model = XGBClassifier(**params)
    
    model.fit(x_train,y_train,
            #   eval_set=[(x_train,y_train),(x_test,y_test)],
              verbose=0,
            #   eval_metric='mlogloss',
            #   early_stopping_rounds=50,
              )
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test,y_predict)
    
    return results
xgb_bo = BayesianOptimization(f=xgb_hamsu,
                              pbounds=xgb_parameters,
                              random_state=123)

xgb_bo.maximize(init_points=2,
                n_iter=10)

print(xgb_bo.max)

# {'target': 0.9824561403508771, 
#  'params': {'colsample_bytree': 0.8608381131350089, 
#             'max_depth': 5.635105053182353, 
#             'min_child_weight': 1.1676876570215784, 
#             'reg_alpha': 46.548028606884, 
#             'reg_lambda': 8.347404889065892, 
#             'subsample': 0.9536812652106865}}

xgb_parameters = {'max_depth': (2, 8),
              'min_child_weight' : (1, 10),
              'subsample' : (0.5, 1),
              'colsample_bytree' : (0.5, 1),
              'reg_alpha' : (25, 75),
              'reg_lambda' : (5, 15)
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
    model = XGBClassifier(**params)
    
    model.fit(x_train,y_train,
            #   eval_set=[(x_train,y_train),(x_test,y_test)],
              verbose=0,
            #   eval_metric='mlogloss',
            #   early_stopping_rounds=50,
              )
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test,y_predict)
    
    return results
xgb_bo = BayesianOptimization(f=xgb_hamsu,
                              pbounds=xgb_parameters,
                              random_state=123)

xgb_bo.maximize(init_points=2,
                n_iter=10)

print(xgb_bo.max)

# {'target': 0.9912280701754386, 
#  'params': {'colsample_bytree': 1.0, 
#             'max_depth': 7.420480427292929, 
#             'min_child_weight': 3.201045873872221,
#             'reg_alpha': 44.48925824178155, 
#             'reg_lambda': 6.029802176489195, 
#             'subsample': 0.9548886503523482}}