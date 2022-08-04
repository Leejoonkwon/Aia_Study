import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC 
from sklearn.svm import LinearSVR 
import matplotlib
from tqdm import tqdm
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
#1. 데이터
datasets= load_breast_cancer()
x = datasets['data']
y = datasets['target']
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

allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='Regressor')

# print('allAlgorithms :',allAlgorithms)
print('모델의 갯수 :',len(allAlgorithms)) #모델의 갯수 : 41

for (name,algorithms) in allAlgorithms:
    try: # for문을 실행하는 와중에 예외 (error)가 발생하면 무시하고 진행 <예외처리>
        model = algorithms()
        model.fit(x_train,y_train)
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test,y_predict)
        scores = cross_val_score(model, x, y, cv=kfold)
        # print('{}의 정확도 {}의 ',name,'의 정답률 :',acc)
        print('{}의 정확도 :{} 검증 평균: {} '.format(name,round(acc,3),round(np.mean(scores),4)))
       
    except:
        continue
        # print(name,'은 안나온 놈!!!')

# AdaBoostClassifier의 정확도 :0.947 검증 평균: 0.9649 
# BaggingClassifier의 정확도 :0.965 검증 평균: 0.9473 
# BernoulliNB의 정확도 :0.947 검증 평균: 0.6274
# CalibratedClassifierCV의 정확도 :0.965 검증 평균: 0.9263 
# DecisionTreeClassifier의 정확도 :0.947 검증 평균: 0.9192 
# DummyClassifier의 정확도 :0.57 검증 평균: 0.6274
# ExtraTreeClassifier의 정확도 :0.921 검증 평균: 0.9191
# ExtraTreesClassifier의 정확도 :0.956 검증 평균: 0.9701 
# GaussianNB의 정확도 :0.939 검증 평균: 0.942
# GaussianProcessClassifier의 정확도 :0.974 검증 평균: 0.9122 
# GradientBoostingClassifier의 정확도 :0.965 검증 평균: 0.9614 
# HistGradientBoostingClassifier의 정확도 :0.956 검증 평균: 0.9737 
# KNeighborsClassifier의 정확도 :0.947 검증 평균: 0.928 
# LabelPropagation의 정확도 :0.956 검증 평균: 0.3902 
# LabelSpreading의 정확도 :0.947 검증 평균: 0.3902 
# LinearDiscriminantAnalysis의 정확도 :0.956 검증 평균: 0.9614       
# LinearSVC의 정확도 :0.965 검증 평균: 0.9157 
# LogisticRegression의 정확도 :0.974 검증 평균: 0.9385 
# LogisticRegressionCV의 정확도 :0.965 검증 평균: 0.9508 
# MLPClassifier의 정확도 :0.982 검증 평균: 0.9245 
# NearestCentroid의 정확도 :0.93 검증 평균: 0.8893
# NuSVC의 정확도 :0.939 검증 평균: 0.8735 
# PassiveAggressiveClassifier의 정확도 :0.974 검증 평균: 0.9122      
# Perceptron의 정확도 :0.956 검증 평균: 0.7771 
# QuadraticDiscriminantAnalysis의 정확도 :0.956 검증 평균: 0.9525    
# RandomForestClassifier의 정확도 :0.965 검증 평균: 0.9596 
# RidgeClassifier의 정확도 :0.956 검증 평균: 0.9543
# RidgeClassifierCV의 정확도 :0.947 검증 평균: 0.9561
# SGDClassifier의 정확도 :0.974 검증 평균: 0.9087 
# SVC의 정확도 :0.965 검증 평균: 0.921 