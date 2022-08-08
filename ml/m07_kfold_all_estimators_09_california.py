
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC 
from sklearn.svm import LinearSVR 
from sklearn.metrics import r2_score
import numpy as np

#1. 데이터
datasets = fetch_california_housing()
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
# ARDRegression의 r2_score :0.6 검증 평균: 0.5921 
# AdaBoostRegressor의 r2_score :0.598 검증 평균: 0.5745 
# BaggingRegressor의 r2_score :0.788 검증 평균: 0.7652 
# BayesianRidge의 r2_score :0.601 검증 평균: 0.588 
# CCA의 r2_score :0.381 검증 평균: 0.2078 
# DecisionTreeRegressor의 r2_score :0.572 검증 평균: 0.5642
# DummyRegressor의 r2_score :-0.005 검증 평균: -0.0082     
# ElasticNet의 r2_score :0.553 검증 평균: 0.5775 
# ElasticNetCV의 r2_score :0.598 검증 평균: 0.5404 
# ExtraTreeRegressor의 r2_score :0.502 검증 평균: 0.558 
# ExtraTreesRegressor의 r2_score :0.82 검증 평균: 0.7969 
# GammaRegressor의 r2_score :0.51 검증 평균: -0.0057 
# GaussianProcessRegressor의 r2_score :0.539 검증 평균: -1.718
# GradientBoostingRegressor의 r2_score :0.792 검증 평균: 0.7713
# HistGradientBoostingRegressor의 r2_score :0.819 검증 평균: 0.7886
# HuberRegressor의 r2_score :0.577 검증 평균: 0.5649 
# KNeighborsRegressor의 r2_score :0.668 검증 평균: 0.3914 
# KernelRidge의 r2_score :-1.084 검증 평균: 0.5893 
# Lars의 r2_score :0.601 검증 평균: 0.5919 
# LarsCV의 r2_score :0.6 검증 평균: 0.5915 
# Lasso의 r2_score :0.597 검증 평균: 0.5834 
# LassoCV의 r2_score :0.599 검증 평균: 0.5716 
# LassoLars의 r2_score :0.277 검증 평균: 0.3023 
# LassoLarsCV의 r2_score :0.6 검증 평균: 0.592 
# LassoLarsIC의 r2_score :0.6 검증 평균: 0.5919 
# LinearRegression의 r2_score :0.601 검증 평균: 0.5919 
# LinearSVR의 r2_score :0.515 검증 평균: -1.4158 
# MLPRegressor의 r2_score :0.608 검증 평균: 0.5752 
# NuSVR의 r2_score :0.44 검증 평균: 0.0413 
# OrthogonalMatchingPursuit의 r2_score :0.353 검증 평균: 0.3737
# OrthogonalMatchingPursuitCV의 r2_score :0.592 검증 평균: 0.58
# PLSCanonical의 r2_score :-0.108 검증 평균: -0.4294 
# PLSRegression의 r2_score :0.589 검증 평균: 0.5851 
# PassiveAggressiveRegressor의 r2_score :0.489 검증 평균: 0.3163
# PoissonRegressor의 r2_score :0.647 검증 평균: -0.008 
# RANSACRegressor의 r2_score :0.48 검증 평균: 0.5093 
# RandomForestRegressor의 r2_score :0.796 검증 평균: 0.7828
# Ridge의 r2_score :0.601 검증 평균: 0.5901 
# RidgeCV의 r2_score :0.601 검증 평균: 0.5918 
# SGDRegressor의 r2_score :0.602 검증 평균: -1.430096769094624e+25
# SVR의 r2_score :0.432 검증 평균: 0.0499 
# TheilSenRegressor의 r2_score :0.581 검증 평균: 0.5755 
# TransformedTargetRegressor의 r2_score :0.601 검증 평균: 0.5919
# TweedieRegressor의 r2_score :0.511 검증 평균: 0.5599    