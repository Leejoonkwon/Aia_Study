
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten,LSTM,Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
from sklearn.svm import LinearSVC 
from sklearn.svm import LinearSVR 

#1. 데이터
path = 'D:\study_data\_data\_csv\kaggle_bike/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
print(train_set)

print(train_set.shape) #(10886, 11)

test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)

sampleSubmission = pd.read_csv(path + 'sampleSubmission.csv',#예측에서 쓸거야!!
                       index_col=0)
            
print(test_set)
print(test_set.shape) #(6493, 8) #train_set과 열 값이 '1'차이 나는 건 count를 제외했기 때문이다.예측 단계에서 값을 대입

print(train_set.columns)
print(train_set.info()) #null은 누락된 값이라고 하고 "결측치"라고도 한다.
print(train_set.describe()) 


###### 결측치 처리 1.제거##### dropna 사용
print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
print(train_set.shape) #(10886,11)


x = train_set.drop([ 'casual', 'registered','count'],axis=1) #axis는 컬럼 


print(x.columns)
print(x.shape) #(10886, 8)

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
import numpy as np

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
    
    
# ARDRegression의 r2_score :0.267 검증 평균: 0.2599 
# AdaBoostRegressor의 r2_score :0.176 검증 평균: 0.2052 
# BaggingRegressor의 r2_score :0.208 검증 평균: 0.234 
# BayesianRidge의 r2_score :0.267 검증 평균: 0.2599 
# CCA의 r2_score :-0.183 검증 평균: -0.1308 
# DecisionTreeRegressor의 r2_score :-0.25 검증 평균: -0.1475 
# DummyRegressor의 r2_score :-0.002 검증 평균: -0.0006
# ElasticNet의 r2_score :0.248 검증 평균: 0.258 
# ElasticNetCV의 r2_score :0.265 검증 평균: 0.2438 
# ExtraTreeRegressor의 r2_score :-0.166 검증 평균: -0.069 
# ExtraTreesRegressor의 r2_score :0.144 검증 평균: 0.1961 
# GammaRegressor의 r2_score :0.219 검증 평균: 0.1762 
# GaussianProcessRegressor의 r2_score :-7713.014 검증 평균: -0.2584 
# GradientBoostingRegressor의 r2_score :0.32 검증 평균: 0.3301 
# HistGradientBoostingRegressor의 r2_score :0.338 검증 평균: 0.3489 
# HuberRegressor의 r2_score :0.251 검증 평균: 0.2358 
# KNeighborsRegressor의 r2_score :0.261 검증 평균: 0.2047 
# KernelRidge의 r2_score :-0.877 검증 평균: 0.2434 
# Lars의 r2_score :0.267 검증 평균: 0.2599 
# LarsCV의 r2_score :0.267 검증 평균: 0.2553 
# Lasso의 r2_score :0.267 검증 평균: 0.2598 
# LassoCV의 r2_score :0.267 검증 평균: 0.2411 
# LassoLars의 r2_score :-0.002 검증 평균: -0.0006 
# LassoLarsCV의 r2_score :0.267 검증 평균: 0.2586 
# LassoLarsIC의 r2_score :0.267 검증 평균: 0.2598 
# LinearRegression의 r2_score :0.267 검증 평균: 0.2599 
# LinearSVR의 r2_score :0.231 검증 평균: 0.2221 
# MLPRegressor의 r2_score :0.302 검증 평균: 0.2878 
# NuSVR의 r2_score :0.237 검증 평균: 0.2159 
# OrthogonalMatchingPursuit의 r2_score :0.151 검증 평균: 0.1549 
# OrthogonalMatchingPursuitCV의 r2_score :0.265 검증 평균: 0.2404 
# PLSCanonical의 r2_score :-0.666 검증 평균: -0.5759 
# PLSRegression의 r2_score :0.259 검증 평균: 0.2535 
# PassiveAggressiveRegressor의 r2_score :0.244 검증 평균: -0.1523 
# PoissonRegressor의 r2_score :0.247 검증 평균: 0.2699 
# RANSACRegressor의 r2_score :-0.095 검증 평균: -0.0213 
# RadiusNeighborsRegressor의 r2_score :-2.2516219283312722e+31 검증 평균: -1.2447906586968304e+33
# RandomForestRegressor의 r2_score :0.244 검증 평균: 0.2875 
# Ridge의 r2_score :0.267 검증 평균: 0.2599 
# RidgeCV의 r2_score :0.267 검증 평균: 0.2599 
# SGDRegressor의 r2_score :0.268 검증 평균: -8.4763456797917e+16 
# SVR의 r2_score :0.226 검증 평균: 0.1976 
# TheilSenRegressor의 r2_score :0.232 검증 평균: 0.2502 
# TransformedTargetRegressor의 r2_score :0.267 검증 평균: 0.2599 
# TweedieRegressor의 r2_score :0.225 검증 평균: 0.2559     
    
      