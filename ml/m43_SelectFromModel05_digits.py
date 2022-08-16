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
datasets = load_digits()
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
# 테스트 스코어:  0.9638888888888889
# acc_score 결과:  0.9160392902344405
# [0.         0.01065247 0.00948314 0.0035076  0.00445116 0.03638005
#  0.00443837 0.         0.         0.00218903 0.03074838 0.00059684
#  0.00161971 0.0335904  0.01468154 0.         0.         0.00158149
#  0.00918587 0.04863956 0.01072436 0.04088914 0.00139157 0.02727327
#  0.         0.00239637 0.0402486  0.00708215 0.01814266 0.04750339
#  0.05919282 0.         0.         0.09873619 0.01948814 0.01332176
#  0.00381819 0.02032899 0.038459   0.         0.         0.0149009
#  0.00687949 0.01813629 0.00848222 0.00592984 0.05046677 0.
#  0.         0.00202224 0.00536231 0.03339408 0.00206126 0.00688344
#  0.00250292 0.         0.         0.         0.01201635 0.00700372
#  0.11105001 0.00364167 0.01470325 0.03382109]
# -----------------------------------------------
# (1437, 64) (1437, 64)
# Thresh=0.000, n=64, R2: 65.56% 

# (1437, 26) (1437, 26)
# Thresh=0.011, n=26, R2: 67.86% 

# (1437, 27) (1437, 27)
# Thresh=0.009, n=27, R2: 64.82% 

# (1437, 40) (1437, 40)
# Thresh=0.004, n=40, R2: 74.55% 

# (1437, 36) (1437, 36)
# Thresh=0.004, n=36, R2: 69.83% 

# (1437, 10) (1437, 10)
# Thresh=0.036, n=10, R2: 54.78% 

# (1437, 37) (1437, 37)
# Thresh=0.004, n=37, R2: 73.04% 

# (1437, 64) (1437, 64)
# Thresh=0.000, n=64, R2: 65.56% 

# (1437, 64) (1437, 64)
# Thresh=0.000, n=64, R2: 65.56% 

# (1437, 43) (1437, 43)
# Thresh=0.002, n=43, R2: 70.36% 

# (1437, 14) (1437, 14)
# Thresh=0.031, n=14, R2: 60.78% 

# (1437, 49) (1437, 49)
# Thresh=0.001, n=49, R2: 72.21% 

# (1437, 46) (1437, 46)
# Thresh=0.002, n=46, R2: 71.63% 

# (1437, 12) (1437, 12)
# Thresh=0.034, n=12, R2: 55.63% 

# (1437, 22) (1437, 22)
# Thresh=0.015, n=22, R2: 71.01% 

# (1437, 64) (1437, 64)
# Thresh=0.000, n=64, R2: 65.56% 

# (1437, 64) (1437, 64)
# Thresh=0.000, n=64, R2: 65.56% 

# (1437, 47) (1437, 47)
# Thresh=0.002, n=47, R2: 71.54% 

# (1437, 28) (1437, 28)
# Thresh=0.009, n=28, R2: 69.08% 

# (1437, 5) (1437, 5)
# Thresh=0.049, n=5, R2: 25.35% 

# (1437, 25) (1437, 25)
# Thresh=0.011, n=25, R2: 71.38% 

# (1437, 7) (1437, 7)
# Thresh=0.041, n=7, R2: 39.59% 

# (1437, 48) (1437, 48)
# Thresh=0.001, n=48, R2: 65.18% 

# (1437, 15) (1437, 15)
# Thresh=0.027, n=15, R2: 58.44% 

# (1437, 64) (1437, 64)
# Thresh=0.000, n=64, R2: 65.56% 

# (1437, 42) (1437, 42)
# Thresh=0.002, n=42, R2: 73.49% 

# (1437, 8) (1437, 8)
# Thresh=0.040, n=8, R2: 53.90% 

# (1437, 30) (1437, 30)
# Thresh=0.007, n=30, R2: 77.61% 

# (1437, 18) (1437, 18)
# Thresh=0.018, n=18, R2: 67.01% 

# (1437, 6) (1437, 6)
# Thresh=0.048, n=6, R2: 38.37% 

# (1437, 3) (1437, 3)
# Thresh=0.059, n=3, R2: 14.89% 

# (1437, 64) (1437, 64)
# Thresh=0.000, n=64, R2: 65.56% 

# (1437, 64) (1437, 64)
# Thresh=0.000, n=64, R2: 65.56% 

# (1437, 2) (1437, 2)
# Thresh=0.099, n=2, R2: 12.58% 

# (1437, 17) (1437, 17)
# Thresh=0.019, n=17, R2: 63.76% 

# (1437, 23) (1437, 23)
# Thresh=0.013, n=23, R2: 68.54% 

# (1437, 38) (1437, 38)
# Thresh=0.004, n=38, R2: 68.50% 

# (1437, 16) (1437, 16)
# Thresh=0.020, n=16, R2: 66.98% 

# (1437, 9) (1437, 9)
# Thresh=0.038, n=9, R2: 48.06% 

# (1437, 64) (1437, 64)
# Thresh=0.000, n=64, R2: 65.56% 

# (1437, 64) (1437, 64)
# Thresh=0.000, n=64, R2: 65.56% 

# (1437, 20) (1437, 20)
# Thresh=0.015, n=20, R2: 70.09% 

# (1437, 33) (1437, 33)
# Thresh=0.007, n=33, R2: 71.97% 

# (1437, 19) (1437, 19)
# Thresh=0.018, n=19, R2: 68.42% 

# (1437, 29) (1437, 29)
# Thresh=0.008, n=29, R2: 68.41% 

# (1437, 34) (1437, 34)
# Thresh=0.006, n=34, R2: 74.39% 

# (1437, 4) (1437, 4)
# Thresh=0.050, n=4, R2: 36.41% 

# (1437, 64) (1437, 64)
# Thresh=0.000, n=64, R2: 65.56% 

# (1437, 64) (1437, 64)
# Thresh=0.000, n=64, R2: 65.56% 

# (1437, 45) (1437, 45)
# Thresh=0.002, n=45, R2: 73.10% 

# (1437, 35) (1437, 35)
# Thresh=0.005, n=35, R2: 76.98% 

# (1437, 13) (1437, 13)
# Thresh=0.033, n=13, R2: 59.13% 

# (1437, 44) (1437, 44)
# Thresh=0.002, n=44, R2: 70.73% 

# (1437, 32) (1437, 32)
# Thresh=0.007, n=32, R2: 73.39% 

# (1437, 41) (1437, 41)
# Thresh=0.003, n=41, R2: 74.46% 

# (1437, 64) (1437, 64)
# Thresh=0.000, n=64, R2: 65.56% 

# (1437, 64) (1437, 64)
# Thresh=0.000, n=64, R2: 65.56% 

# (1437, 64) (1437, 64)
# Thresh=0.000, n=64, R2: 65.56% 

# (1437, 24) (1437, 24)
# Thresh=0.012, n=24, R2: 71.77% 

# (1437, 31) (1437, 31)
# Thresh=0.007, n=31, R2: 75.44% 

# (1437, 1) (1437, 1)
# Thresh=0.111, n=1, R2: 7.19% 

# (1437, 39) (1437, 39)
# Thresh=0.004, n=39, R2: 71.31% 

# (1437, 21) (1437, 21)
# Thresh=0.015, n=21, R2: 65.04% 

# (1437, 11) (1437, 11)
# Thresh=0.034, n=11, R2: 53.91% 




