from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_diabetes()
x, y = datasets.data, datasets.target

x_train,x_test,y_train,y_test = train_test_split(
    x,y, train_size=0.8,random_state=123
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



# {'target': 0.6127348497750551, 
#  'params': {'colsample_bytree': 0.5, 
#             'max_bin': 163.43634572569303, 
#             'max_depth': 6.0, 
#             'min_child_sample': 10.0, 
#             'min_child_weight': 27.588120195833895, 
#             'num_leaves': 41.93359609407643, 
#             'reg_alpha': 0.01, 
#             'reg_lambda': 0.001, 
#             'subsample': 1.0}

#2. 모델
bayesian_params = {
    'max_depth' : (3, 9),
    'num_leaves': (24,64),
    'min_child_sample' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (10, 200),
    'reg_lambda' : (0.001, 10),
    'reg_alpha' : (0.01, 50),
        }

def lgb_hamsu(max_depth,num_leaves,min_child_sample,min_child_weight,
              subsample,colsample_bytree,max_bin,reg_lambda,reg_alpha) :
    params = {
        'n_estimators':500,"learning_rate":0.01,
        'max_depth' : int(round(max_depth)),  #무조건 정수
        'num_leaves' : int(round(num_leaves)), #무조건 정수
        'min_child_samples' : int(round(min_child_sample)),
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample,1),0), # 0~1 사이의 값이 들어가도록 한다.
        'colsample_bytree' : max(min(colsample_bytree,1),0),
        'max_bin' : max(int(round(max_bin)),10), #무조건 10이상만 
        'reg_lambda' : max(reg_lambda, 0), # 무조건 양수만 
        'reg_alpha' : max(reg_alpha, 0),
          }
    
    # *여러개의 인자를 받겠다.
    # **키워드 받겠다(딕셔너리형태)
    model = LGBMRegressor(**params)
    
    model.fit(x_train,y_train,
              eval_set=[(x_train,y_train),(x_test,y_test)],
              verbose=0,
              eval_metric='rmse',
              early_stopping_rounds=50,
              )
    y_predict = model.predict(x_test)
    results = r2_score(y_test,y_predict)
    
    return results
lgb_bo = BayesianOptimization(f=lgb_hamsu,
                              pbounds=bayesian_params,
                              random_state=123)
lgb_bo.maximize(init_points=5,
                n_iter=100)
print(lgb_bo.max)

######################### [실습] ###################
#1. 수정한 파라미터로 모델 만들어서 비교!!!! 

#2. 수정한 파라미터를 이용해서 파라미터 재조정!!! 
