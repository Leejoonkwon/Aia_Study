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
datasets = fetch_california_housing()
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

# 테스트 스코어:  0.7829929935589588
# acc_score 결과:  0.7829929935589588
# [0.24213526 0.04204758 0.25923392 0.03098403 0.01171169 0.16838305
#  0.14919013 0.09631437]
# -----------------------------------------------
# (16512, 2) (16512, 2)
# Thresh=0.242, n=2, R2: 52.37% 

# (16512, 6) (16512, 6)
# Thresh=0.042, n=6, R2: 78.97% 

# (16512, 1) (16512, 1)
# Thresh=0.259, n=1, R2: 14.08% 

# (16512, 7) (16512, 7)
# Thresh=0.031, n=7, R2: 76.82% 

# (16512, 8) (16512, 8)
# Thresh=0.012, n=8, R2: 78.30% 

# (16512, 3) (16512, 3)
# Thresh=0.168, n=3, R2: 59.54% 

# (16512, 4) (16512, 4)
# Thresh=0.149, n=4, R2: 68.61% 

# (16512, 5) (16512, 5)
# Thresh=0.096, n=5, R2: 77.42% 


