import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import pandas as pd 
#1.데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(581012, 54) (581012,)
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,KFold,StratifiedKFold

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.8
       ,random_state=1234,shuffle=True)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train) 
# x_test = scaler.transform(x_test)

#2. 모델 
from sklearn.svm import LinearSVC, SVC 
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

from sklearn.pipeline import make_pipeline,Pipeline 

# model = SVC()
# model = make_pipeline(MinMaxScaler(),SVC())
# model = make_pipeline(StandardScaler(),RandomForestClassifier())
pipe = Pipeline([('minmax',MinMaxScaler()),
                  ('RF',RandomForestClassifier())
                  ])
#
# 모델 정의와 스케일링을 정의해주지 않아도  fit에서 fit_transform이 적용된다.
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=100)

parameters = [
    {'RF__n_estimators':[100, 200],'RF__max_depth':[6, 8],'RF__min_samples_leaf':[3,5],
     'RF__min_samples_split':[2, 3],'RF__n_jobs':[-1, 2]},
    {'RF__n_estimators':[300, 400],'RF__max_depth':[6, 8],'RF__min_samples_leaf':[7, 10],
     'RF__min_samples_split':[4, 7],'RF__n_jobs':[-1, 4]}
   
    ]     
model = HalvingGridSearchCV(pipe,parameters,cv = kfold,refit=True,n_jobs=-1,verbose=1)

# Fitting 5 folds(kfold의 인수) for each of 42 candidates, totalling 210 fits(42*5)
# n_jobs=-1 사용할 CPU 갯수를 지정하는 옵션 '-1'은 최대 갯수를 쓰겠다는 뜻
#3. 컴파일,훈련
import time
start = time.time()
model.fit(x_train,y_train) 
end = time.time()- start


# #4.  평가,예측
# results = model.score(x_test,y_test) #분류 모델과 회귀 모델에서 score를 쓰면 알아서 값이 나온다 
# print("results :",results)
# # results : 0.9736842105263158

print("최적의 매개변수 :",model.best_estimator_)
print("최적의 파라미터 :",model.best_params_)
print("best_score :",model.best_score_)
print("model_score :",model.score(x_test,y_test))
y_predict = model.predict(x_test)
print('accuracy_score :',accuracy_score(y_test,y_predict))
y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠  ACC :',accuracy_score(y_test,y_predict))
print("걸린 시간 :",round(end,2),"초")

#============= pipe HalvingGridSearchCV
