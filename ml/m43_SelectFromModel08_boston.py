import numpy as np
from sklearn.datasets import load_iris,load_boston,load_digits,load_breast_cancer
from sklearn.datasets import load_boston,fetch_covtype,fetch_california_housing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel

# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=1234, train_size=0.8)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = XGBRegressor(n_estimators=100,
              learning_rate=1,
              max_depth=2,
              gamma=0,
              min_child_weight=1,
              subsample=1,
              colsample_bytree=0.5,
              colsample_bylevel=1,
              colsample_bynode=1,
              reg_alpha=0.01,
              reg_lambd=1,
              tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, random_state=1234,
              )

model.fit(x_train, y_train, early_stopping_rounds=10, 
          eval_set=[(x_train, y_train), (x_test, y_test)]
          )

print('테스트 스코어: ', model.score(x_test, y_test))

r2 = r2_score(y_test, model.predict(x_test))
print('acc_score 결과: ', r2)

print(model.feature_importances_)
# [0.02737029 0.06044502 0.2727516  0.07338018 0.02401855 0.06909694
#  0.03971948 0.24999845 0.07974161 0.10347791]

thresholds = model.feature_importances_
print('-----------------------------------------------')
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_train.shape)
    
    selection_model = XGBRegressor(n_estimators=100,
              learning_rate=1,
              max_depth=2,
              gamma=0,
              min_child_weight=1,
              subsample=1,
              colsample_bytree=0.5,
              colsample_bylevel=1,
              colsample_bynode=1,
              reg_alpha=0.01,
              tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, random_state=1234,
              )
    
    selection_model.fit(select_x_train, y_train)
    
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    print('Thresh=%.3f, n=%d, R2: %.2f%%'%(thresh, select_x_train.shape[1], score*100), '\n')

# 테스트 스코어:  0.8208776710793739
# acc_score 결과:  0.8208776710793739
# [0.05517107 0.00852136 0.03864899 0.02511239 0.0194922  0.0895038
#  0.03961516 0.0156719  0.01996997 0.07966852 0.06927608 0.04030788
#  0.49904066]
# -----------------------------------------------
# (404, 5) (404, 5)
# Thresh=0.055, n=5, R2: 78.22% 

# (404, 13) (404, 13)
# Thresh=0.009, n=13, R2: 85.11% 

# (404, 8) (404, 8)
# Thresh=0.039, n=8, R2: 87.92% 

# (404, 9) (404, 9)
# Thresh=0.025, n=9, R2: 87.92% 

# (404, 11) (404, 11)
# Thresh=0.019, n=11, R2: 87.75% 

# (404, 2) (404, 2)
# Thresh=0.090, n=2, R2: 69.28% 

# (404, 7) (404, 7)
# Thresh=0.040, n=7, R2: 81.71% 

# (404, 12) (404, 12)
# Thresh=0.016, n=12, R2: 82.85% 

# (404, 10) (404, 10)
# Thresh=0.020, n=10, R2: 81.60% 

# (404, 3) (404, 3)
# Thresh=0.080, n=3, R2: 72.70% 

# (404, 4) (404, 4)
# Thresh=0.069, n=4, R2: 84.08% 

# (404, 6) (404, 6)
# Thresh=0.040, n=6, R2: 79.13% 

# (404, 1) (404, 1)
# Thresh=0.499, n=1, R2: 39.41% 




