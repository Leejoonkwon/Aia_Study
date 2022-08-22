import numpy as np
from sklearn.datasets import load_iris,load_boston,load_digits,load_breast_cancer,load_wine
from sklearn.datasets import load_boston,fetch_covtype,fetch_california_housing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel

# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=1234, train_size=0.8)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = XGBClassifier(n_estimators=100,
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
# 테스트 스코어:  0.9722222222222222
# acc_score 결과:  0.9472913616398243
# [0.01169864 0.05024325 0.00171308 0.00264352 0.         0.03281694
#  0.05708249 0.00214115 0.00233366 0.30660474 0.24723305 0.04431141
#  0.24117804]
# (142, 8) (142, 8)
# Thresh=0.012, n=8, R2: 75.82% 

# (142, 5) (142, 5)
# Thresh=0.050, n=5, R2: 81.58% 

# (142, 12) (142, 12)
# Thresh=0.002, n=12, R2: 68.30% 

# (142, 9) (142, 9)
# Thresh=0.003, n=9, R2: 82.83% 

# (142, 13) (142, 13)
# Thresh=0.000, n=13, R2: 86.15% 

# (142, 7) (142, 7)
# Thresh=0.033, n=7, R2: 79.49% 

# (142, 4) (142, 4)
# Thresh=0.057, n=4, R2: 89.34% 

# (142, 11) (142, 11)
# Thresh=0.002, n=11, R2: 85.55% 

# (142, 10) (142, 10)
# Thresh=0.002, n=10, R2: 82.20% 

# (142, 1) (142, 1)
# Thresh=0.307, n=1, R2: -9.30% 

# (142, 2) (142, 2)
# Thresh=0.247, n=2, R2: 10.08% 

# (142, 6) (142, 6)
# Thresh=0.044, n=6, R2: 89.35% 

# (142, 3) (142, 3)
# Thresh=0.241, n=3, R2: 48.51% 



