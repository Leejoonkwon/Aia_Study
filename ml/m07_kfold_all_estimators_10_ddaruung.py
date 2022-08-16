#  과제
# activation : sigmoid,relu,linear
# metrics 추가
# EarlyStopping  넣고
# 성능비교
# 감상문 2줄이상!
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten,LSTM,Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
import pandas as pd
from sklearn.svm import LinearSVC 
from sklearn.svm import LinearSVR 
#1. 데이터
path = 'D:\study_data\_data\_csv\_ddarung/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
print(train_set)

print(train_set.shape) #(1459, 10)

test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)
submission = pd.read_csv(path + 'submission.csv',#예측에서 쓸거야!!
                       index_col=0)
                       
print(test_set)
print(test_set.shape) #(715, 9) #train_set과 열 값이 '1'차이 나는 건 count를 제외했기 때문이다.예측 단계에서 값을 대입

print(train_set.columns)
print(train_set.info()) #null은 누락된 값이라고 하고 "결측치"라고도 한다.
print(train_set.describe()) 

###### 결측치 처리 1.제거##### dropna 사용
print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
train_set = train_set.fillna(train_set.median())
print(train_set.isnull().sum())
print(train_set.shape)
test_set = test_set.fillna(test_set.median())

x = train_set.drop(['count'],axis=1) #axis는 컬럼 
print(x.columns)
print(x.shape) #(1459, 9)

y = train_set['count']
from sklearn.model_selection import KFold,cross_val_score,cross_val_predict
import warnings 
from sklearn.metrics import r2_score
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
import numpy as np
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
# AdaBoostRegressor의 r2_score :0.577 검증 평균: 0.5798 
# BaggingRegressor의 r2_score :0.75 검증 평균: 0.7662 
# BayesianRidge의 r2_score :0.601 검증 평균: 0.588 
# CCA의 r2_score :0.381 검증 평균: 0.2078 
# DecisionTreeRegressor의 r2_score :0.534 검증 평균: 0.5816 
# DummyRegressor의 r2_score :-0.005 검증 평균: -0.0082
# ElasticNet의 r2_score :0.553 검증 평균: 0.5775 
# ElasticNetCV의 r2_score :0.598 검증 평균: 0.5404 
# ExtraTreeRegressor의 r2_score :0.57 검증 평균: 0.5775 
# ExtraTreesRegressor의 r2_score :0.816 검증 평균: 0.7987 
# GammaRegressor의 r2_score :0.51 검증 평균: -0.0057 
# GaussianProcessRegressor의 r2_score :0.539 검증 평균: -1.718 
# GradientBoostingRegressor의 r2_score :0.792 검증 평균: 0.7711 
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
# LinearSVR의 r2_score :0.517 검증 평균: -0.1638 
# MLPRegressor의 r2_score :0.604 검증 평균: 0.5962 
# NuSVR의 r2_score :0.44 검증 평균: 0.0413 
# OrthogonalMatchingPursuit의 r2_score :0.353 검증 평균: 0.3737 
# OrthogonalMatchingPursuitCV의 r2_score :0.592 검증 평균: 0.58 
# PLSCanonical의 r2_score :-0.108 검증 평균: -0.4294 
# PLSRegression의 r2_score :0.589 검증 평균: 0.5851 
# PassiveAggressiveRegressor의 r2_score :0.562 검증 평균: 0.4072 
# PoissonRegressor의 r2_score :0.647 검증 평균: -0.008 
# RANSACRegressor의 r2_score :0.48 검증 평균: 0.4897 
# RandomForestRegressor의 r2_score :0.793 검증 평균: 0.7785 
# Ridge의 r2_score :0.601 검증 평균: 0.5901 
# RidgeCV의 r2_score :0.601 검증 평균: 0.5918 
# SGDRegressor의 r2_score :0.6 검증 평균: -2.170649858462154e+25 
# SVR의 r2_score :0.432 검증 평균: 0.0499 
# TheilSenRegressor의 r2_score :0.585 검증 평균: 0.5738 
# TransformedTargetRegressor의 r2_score :0.601 검증 평균: 0.5919    
TweedieRegressor의 r2_score :0.511 검증 평균: 0.5599     
    
>>>>>>> 2c957d9ba7d95ae2a556526cd2109205d9849f68
    