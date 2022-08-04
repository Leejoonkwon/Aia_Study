#2. 모델 구성
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import accuracy_score,r2_score
import numpy as np
# matplotlib.rcParams['font.family']='Malgun Gothic'
# matplotlib.rcParams['axes.unicode_minus']=False
import time
from sklearn.svm import LinearSVC

#1. 데이터
datasets = load_boston()
x = datasets.data #데이터를 리스트 형태로 불러올 때 함
y = datasets.target
from sklearn.model_selection import KFold,cross_val_score,cross_val_predict
import warnings 
warnings.filterwarnings('ignore')
x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.8,shuffle=True,random_state=100)
n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)
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
# ARDRegression의 r2_score :0.754 검증 평균: 0.6985 
# AdaBoostRegressor의 r2_score :0.813 검증 평균: 0.8424 
# BaggingRegressor의 r2_score :0.862 검증 평균: 0.8597 
# BayesianRidge의 r2_score :0.752 검증 평균: 0.7038 
# CCA의 r2_score :0.767 검증 평균: 0.6471
# DecisionTreeRegressor의 r2_score :0.662 검증 평균: 0.748 
# DummyRegressor의 r2_score :-0.002 검증 평균: -0.0135     
# ElasticNet의 r2_score :0.632 검증 평균: 0.6708
# ElasticNetCV의 r2_score :0.751 검증 평균: 0.6565 
# ExtraTreeRegressor의 r2_score :0.499 검증 평균: 0.6991 
# ExtraTreesRegressor의 r2_score :0.887 검증 평균: 0.8763 
# GammaRegressor의 r2_score :0.623 검증 평균: -0.0136      
# GaussianProcessRegressor의 r2_score :0.391 검증 평균: -5.9286
# GradientBoostingRegressor의 r2_score :0.904 검증 평균: 0.8859
# HistGradientBoostingRegressor의 r2_score :0.86 검증 평균: 0.8581
# HuberRegressor의 r2_score :0.728 검증 평균: 0.5916 
# KNeighborsRegressor의 r2_score :0.791 검증 평균: 0.5286 
# KernelRidge의 r2_score :-4.583 검증 평균: 0.6854 
# Lars의 r2_score :0.741 검증 평균: 0.6977 
# LarsCV의 r2_score :0.742 검증 평균: 0.6928 
# Lasso의 r2_score :0.66 검증 평균: 0.6657
# LassoCV의 r2_score :0.754 검증 평균: 0.6779 
# LassoLars의 r2_score :-0.002 검증 평균: -0.0135
# LassoLarsCV의 r2_score :0.753 검증 평균: 0.6965 
# LassoLarsIC의 r2_score :0.756 검증 평균: 0.713
# LinearRegression의 r2_score :0.756 검증 평균: 0.7128 
# LinearSVR의 r2_score :0.713 검증 평균: 0.5318 
# MLPRegressor의 r2_score :0.717 검증 평균: 0.5042 
# NuSVR의 r2_score :0.622 검증 평균: 0.2295 
# OrthogonalMatchingPursuit의 r2_score :0.545 검증 평균: 0.5343
# OrthogonalMatchingPursuitCV의 r2_score :0.698 검증 평균: 0.6578
# PLSCanonical의 r2_score :-1.885 검증 평균: -2.2096       
# PLSRegression의 r2_score :0.721 검증 평균: 0.6847 
# PassiveAggressiveRegressor의 r2_score :0.607 검증 평균: -0.1472
# PoissonRegressor의 r2_score :0.802 검증 평균: 0.7555 
# RANSACRegressor의 r2_score :0.078 검증 평균: 0.4622 
# RandomForestRegressor의 r2_score :0.888 검증 평균: 0.8771
# Ridge의 r2_score :0.755 검증 평균: 0.7109
# RidgeCV의 r2_score :0.751 검증 평균: 0.7128
# SGDRegressor의 r2_score :0.751 검증 평균: -5.055682951474623e+26
# SVR의 r2_score :0.637 검증 평균: 0.1963 
# TheilSenRegressor의 r2_score :0.678 검증 평균: 0.6786 
# TransformedTargetRegressor의 r2_score :0.756 검증 평균: 0.7128
# TweedieRegressor의 r2_score :0.617 검증 평균: 0.6535