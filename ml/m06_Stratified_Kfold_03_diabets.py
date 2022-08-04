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

from sklearn.model_selection import KFold,cross_val_score,cross_val_predict
import warnings 
warnings.filterwarnings('ignore')
x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.8,shuffle=True,random_state=100)
n_split = 5
from sklearn.model_selection import train_test_split,KFold,cross_val_score,StratifiedKFold
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=66)

# kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)
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

print(x_test.shape)

#2. 모델 구성
from sklearn.utils import all_estimators

allAlgorithms = all_estimators(type_filter='regressor')
# allAlgorithms = all_estimators(type_filter='Regressor')

# print('allAlgorithms :',allAlgorithms)
print('모델의 갯수 :',len(allAlgorithms)) #모델의 갯수 : 41

for (name,algorithms) in allAlgorithms:
    try: # for문을 실행하는 와중에 예외 (error)가 발생하면 무시하고 진행 <예외처리>
        model = algorithms()
        model.fit(x_train,y_train)
        
        y_predict = model.predict(x_test)
        # acc = accuracy_score(y_test,y_predict)
        r2  = r2_score(y_test,y_predict)
        scores = cross_val_score(model, x, y, cv=kfold)
        # print('{}의 정확도 {}의 ',name,'의 정답률 :',acc)
        print('{}의 r2_score :{} 검증 평균: {} '.format(name,round(r2,3),round(np.mean(scores),4)))
       
    except:
        continue
    
# ARDRegression의 r2_score :0.501 검증 평균: 0.4859 
# AdaBoostRegressor의 r2_score :0.429 검증 평균: 0.4292 
# BaggingRegressor의 r2_score :0.431 검증 평균: 0.3998 
# BayesianRidge의 r2_score :0.507 검증 평균: 0.4842 
# CCA의 r2_score :0.483 검증 평균: 0.4521
# DecisionTreeRegressor의 r2_score :-0.118 검증 평균: -0.0421
# DummyRegressor의 r2_score :-0.024 검증 평균: -0.0081     
# ElasticNet의 r2_score :0.5 검증 평균: 0.0007 
# ElasticNetCV의 r2_score :0.509 검증 평균: 0.4369 
# ExtraTreeRegressor의 r2_score :-0.012 검증 평균: -0.0609 
# ExtraTreesRegressor의 r2_score :0.391 검증 평균: 0.4354 
# GammaRegressor의 r2_score :0.484 검증 평균: -0.0015      
# GaussianProcessRegressor의 r2_score :-0.698 검증 평균: -12.3008
# GradientBoostingRegressor의 r2_score :0.344 검증 평균: 0.4092
# HistGradientBoostingRegressor의 r2_score :0.412 검증 평균: 0.3656
# HuberRegressor의 r2_score :0.5 검증 평균: 0.4813 
# KNeighborsRegressor의 r2_score :0.318 검증 평균: 0.3677  
# KernelRidge의 r2_score :-3.86 검증 평균: -3.5659 
# Lars의 r2_score :0.504 검증 평균: -0.0492 
# LarsCV의 r2_score :0.491 검증 평균: 0.4033 
# Lasso의 r2_score :0.503 검증 평균: 0.3478
# LassoCV의 r2_score :0.498 검증 평균: 0.4817 
# LassoLars의 r2_score :0.382 검증 평균: 0.3719
# LassoLarsCV의 r2_score :0.496 검증 평균: 0.4809 
# LassoLarsIC의 r2_score :0.491 검증 평균: 0.4865 
# LinearRegression의 r2_score :0.504 검증 평균: 0.4835     
# LinearSVR의 r2_score :0.372 검증 평균: -0.3782
# MLPRegressor의 r2_score :-0.882 검증 평균: -3.093 
# NuSVR의 r2_score :0.177 검증 평균: 0.1576 
# OrthogonalMatchingPursuit의 r2_score :0.363 검증 평균: 0.3035
# OrthogonalMatchingPursuitCV의 r2_score :0.447 검증 평균: 0.4822
# PLSCanonical의 r2_score :-1.642 검증 평균: -1.2346       
# PLSRegression의 r2_score :0.516 검증 평균: 0.4812        
# PassiveAggressiveRegressor의 r2_score :0.508 검증 평균: 0.4733
# PoissonRegressor의 r2_score :0.497 검증 평균: 0.3318 
# RANSACRegressor의 r2_score :-0.022 검증 평균: 0.0424 
# RandomForestRegressor의 r2_score :0.434 검증 평균: 0.4368
# Ridge의 r2_score :0.505 검증 평균: 0.4186
# RidgeCV의 r2_score :0.507 검증 평균: 0.4842 
# SGDRegressor의 r2_score :0.504 검증 평균: 0.4055 
# SVR의 r2_score :0.197 검증 평균: 0.1585 
# TheilSenRegressor의 r2_score :0.49 검증 평균: 0.474 
# TransformedTargetRegressor의 r2_score :0.504 검증 평균: 0.4835
# TweedieRegressor의 r2_score :0.474 검증 평균: -0.0016     
