from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
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
                  ('RF',RandomForestRegressor())
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
model = RandomizedSearchCV(pipe,parameters,cv = kfold,refit=True,n_jobs=-1,verbose=1)

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

#============= pipe HalvingGridSearchCV
# 최적의 매개변수 : Pipeline(steps=[('minmax', MinMaxScaler()),
#                 ('RF',
#                  RandomForestRegressor(max_depth=8, min_samples_leaf=5,
#                                        n_estimators=200, n_jobs=-1))])
# 최적의 파라미터 : {'RF__max_depth': 8, 'RF__min_samples_leaf': 5, 'RF__min_samples_split': 2, 
# 'RF__n_estimators': 200, 'RF__n_jobs': -1}     
# best_score : 0.43426171134831504
# model_score : 0.434176564841507
# accuracy_score : 0.434176564841507
# 최적 튠  ACC : 0.434176564841507
# 걸린 시간 : 26.16 초
#============= pipe GridSearchCV
# 최적의 매개변수 : Pipeline(steps=[('minmax', MinMaxScaler()),
#                 ('RF',
#                  RandomForestRegressor(max_depth=8, min_samples_leaf=5,
#                                        min_samples_split=3, n_jobs=2))])
# 최적의 파라미터 : {'RF__max_depth': 8, 'RF__min_samples_leaf': 5, 'RF__min_samples_split': 3, 
# 'RF__n_estimators': 100, 'RF__n_jobs': 2}      
# best_score : 0.44918741376560345
# model_score : 0.44511231464491186
# accuracy_score : 0.44511231464491186
# 최적 튠  ACC : 0.44511231464491186
# 걸린 시간 : 24.68 초
#============= pipe RandomizedSearchCV
# 최적의 매개변수 : Pipeline(steps=[('minmax', MinMaxScaler()),
#                 ('RF',
#                  RandomForestRegressor(max_depth=8, min_samples_leaf=7,
#                                        min_samples_split=7, n_estimators=300,
#                                        n_jobs=-1))])
# 최적의 파라미터 : {'RF__n_jobs': -1, 'RF__n_estimators': 300, 'RF__min_samples_split': 7, 'RF__min_samples_leaf': 7, 'RF__max_depth': 8}     
# best_score : 0.4482943685050829
# model_score : 0.44276486311307006
# accuracy_score : 0.4427648631130702
# 최적 튠  ACC : 0.4427648631130702
# 걸린 시간 : 4.88 초
