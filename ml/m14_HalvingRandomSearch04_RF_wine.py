import numpy as np
from sklearn.datasets import load_wine
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
#1.데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']

from sklearn.model_selection import KFold,cross_val_score,cross_val_predict

x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.8,shuffle=True,random_state=100)

kfold = KFold(n_splits=5, shuffle=True, random_state=66)
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier #공부하자 
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,StratifiedKFold
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=100)

parameters = [
    {'n_estimators':[100, 200],'max_depth':[6, 8],'min_samples_leaf':[3,5],
     'min_samples_split':[2, 3],'n_jobs':[-1, 2]},
    {'n_estimators':[300, 400],'max_depth':[6, 8],'min_samples_leaf':[7, 10],
     'min_samples_split':[4, 7],'n_jobs':[-1, 4]}
   
    ]     

# 각 횟수를 병렬로 진행해 총 42번을  1회에 한다.
#rbf= Gaussian basis function RBF 뉴럴네트워크의 경우 각 데이터에 맞는 
# Kernel function을 이용하기에 비선형적이고, MLP보다 학습이 빠르다.

#2. 모델 구성



model = HalvingRandomSearchCV(RandomForestClassifier(),parameters,cv=kfold,verbose=1,
                     refit=True,n_jobs=-1) 
# Fitting 5 folds(kfold의 인수) for each of 42 candidates, totalling 210 fits(42*5)
# n_jobs=-1 사용할 CPU 갯수를 지정하는 옵션 '-1'은 최대 갯수를 쓰겠다는 뜻

#3. 컴파일,훈련
import time
start = time.time()
model.fit(x_train,y_train) 
end = time.time()- start


# #4.  평가,예측


print("최적의 매개변수 :",model.best_estimator_)
print("최적의 파라미터 :",model.best_params_)
print("best_score :",model.best_score_)
print("model_score :",model.score(x_test,y_test))
y_predict = model.predict(x_test)
print('accuracy_score :',accuracy_score(y_test,y_predict))
y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠  ACC :',accuracy_score(y_test,y_predict))
print("걸린 시간 :",round(end,2),"초")
#=============gridsearch
# 최적의 매개변수 : RandomForestClassifier(max_depth=8, min_samples_leaf=3, n_jobs=2)
# 최적의 파라미터 : {'max_depth': 8, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 100, 'n_jobs': 2}   
# best_score : 0.9582417582417582
# model_score : 0.9649122807017544
# accuracy_score : 0.9649122807017544
# 최적 튠  ACC : 0.9649122807017544
#=============randomsearch
# 최적의 매개변수 : RandomForestClassifier(max_depth=8, min_samples_leaf=3, n_estimators=200,
#                        n_jobs=2)
# 최적의 파라미터 : {'n_jobs': 2, 'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 3, 'max_depth': 8}   
# best_score : 0.9538461538461538
# model_score : 0.9824561403508771
# accuracy_score : 0.9824561403508771
# 최적 튠  ACC : 0.9824561403508771
# 걸린 시간 : 4.24 초
#=============halvinggridsearch
# 최적의 매개변수 : RandomForestClassifier(max_depth=8, min_samples_leaf=3, n_jobs=-1)
# 최적의 파라미터 : {'max_depth': 8, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 100, 'n_jobs': -1}  
# best_score : 0.9444444444444444
# model_score : 0.9649122807017544
# accuracy_score : 0.9649122807017544
# 최적 튠  ACC : 0.9649122807017544
# 걸린 시간 : 21.36 초
#=============halvingrandomsearch
# 최적의 매개변수 : RandomForestClassifier(max_depth=8, min_samples_leaf=5, min_samples_split=3,
#                        n_estimators=200, n_jobs=2)
# 최적의 파라미터 : {'n_jobs': 2, 'n_estimators': 200, 'min_samples_split': 3, 'min_samples_leaf': 5, 'max_depth': 8}   
# best_score : 0.9777777777777779
# model_score : 1.0
# accuracy_score : 1.0
# 최적 튠  ACC : 1.0
# 걸린 시간 : 3.86 초

