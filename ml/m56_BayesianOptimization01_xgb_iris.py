from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
from sklearn.datasets import load_diabetes,load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,accuracy_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_iris()
x, y = datasets.data, datasets.target

x_train,x_test,y_train,y_test = train_test_split(
    x,y, train_size=0.8,random_state=123
)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)





# parameters = {'n_estimators':[100], # 디폴트 100/ 1~inf 무한대 
# eta[기본값=0.3, 별칭: learning_rate]
# gamma[기본값=0, 별칭: min_split_loss]
# max_depth[기본값=6]
# min_child_weight[기본값=1] 0~inf
# subsample[기본값=1] 0~1
# colsample_bytree [0,0.1,0.2,0.3,0.5,0.7,1]    [기본값=1] 0~1
# colsample_bylevel': [0,0.1,0.2,0.3,0.5,0.7,1] [기본값=1] 0~1
# 'colsample_bynode': [0,0.1,0.2,0.3,0.5,0.7,1] [기본값=1] 0~1
# 'reg_alpha' : [0,0.1 ,0.01, 0.001, 1 ,2 ,10]  [기본값=0] 0~inf /L1 절댓값 가중치 규제 
# 'reg_lambda' : [0,0.1 ,0.01, 0.001, 1 ,2 ,10]  [기본값=1] 0~inf /L2 절댓값 가중치 규제 
# max_delta_step[기본값=0]

from xgboost import XGBClassifier,XGBRegressor
# bayesian_params = {
#     'max_depth' : (3, 9),
#     'num_leaves': (24,64),
#     'min_child_sample' : (10, 200),
#     'min_child_weight' : (1, 50),
#     'subsample' : (0.5, 1),
#     'colsample_bytree' : (0.5, 1),
#     'max_bin' : (10, 200),
#     'reg_lambda' : (0.001, 10),
#     'reg_alpha' : (0.01, 50),
#         }
#2. 모델
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
              eval_set=[(x_train,y_train),(x_test,y_test)],
              verbose=0,
              eval_metric='mlogloss',
              early_stopping_rounds=50,
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

# ######################### [실습] ###################
# #1. 수정한 파라미터로 모델 만들어서 비교!!!! 
{'target': 1.0, 'params': {'colsample_bytree': 2.0, 'max_depth': 7.0, 'min_child_weight': 1.0, 'reg_alpha': 0.01, 'reg_lambda': 10.0, 'subsample': 1.0}}
xgb_parameters = {'max_depth': (1, 7),
              'min_child_weight' : (1, 10),
              'subsample' : (0.1, 1),
              'colsample_bytree' : (0.7, 2),
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
              eval_set=[(x_train,y_train),(x_test,y_test)],
              verbose=0,
              eval_metric='mlogloss',
              early_stopping_rounds=50,
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
#2. 수정한 파라미터를 이용해서 파라미터 재조정!!! 
