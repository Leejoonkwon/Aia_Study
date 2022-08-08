from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,StratifiedKFold
from sklearn.ensemble import RandomForestRegressor #공부하자 

x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.8,shuffle=True,random_state=100)
kfold = KFold(n_splits=5, shuffle=True, random_state=66)
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


parameters = [
    {'n_estimators':[100, 200],'max_depth':[6, 8],'min_samples_leaf':[3,5],
     'min_samples_split':[2, 3],'n_jobs':[-1, 2]},
    {'n_estimators':[300, 400],'max_depth':[6, 8],'min_samples_leaf':[7, 10],
     'min_samples_split':[4, 7],'n_jobs':[-1, 4]}
    ]     

#2. 모델 구성
model = GridSearchCV(RandomForestRegressor(),parameters,cv=kfold,verbose=1,
                     refit=True,n_jobs=-1) 

#3. 컴파일,훈련
import time
start = time.time()
model.fit(x_train,y_train) 
end = time.time()- start

print("최적의 매개변수 :",model.best_estimator_)
print("최적의 파라미터 :",model.best_params_)
print("best_score :",model.best_score_)
print("model_score :",model.score(x_test,y_test))
y_predict = model.predict(x_test)
print('accuracy_score :',r2_score(y_test,y_predict))
y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠  ACC :',r2_score(y_test,y_predict))
print("걸린 시간 :",round(end,2),"초")
# 최적의 매개변수 : RandomForestRegressor(max_depth=8, min_samples_leaf=10, min_samples_split=4,
#                       n_estimators=300, n_jobs=-1)
# 최적의 파라미터 : {'max_depth': 8, 'min_samples_leaf': 10, 
# 'min_samples_split': 4, 'n_estimators': 300, 'n_jobs': -1} 
# best_score : 0.4394970272500867
# model_score : 0.4575146739404726
# accuracy_score : 0.4575146739404726
# 최적 튠  ACC : 0.4575146739404726
# 걸린 시간 : 16.68 초