import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,cross_val_score,cross_val_predict
import warnings 
warnings.filterwarnings('ignore')
#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets['target']
x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.8,shuffle=True,random_state=100)
#2. 모델 구성
from sklearn.utils import all_estimators
n_split = 10
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)
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
# AdaBoostClassifier의 정확도 :0.967 검증 평균: 0.8867 
# BaggingClassifier의 정확도 :0.967 검증 평균: 0.9533 
# BernoulliNB의 정확도 :0.2 검증 평균: 0.2933
# CalibratedClassifierCV의 정확도 :1.0 검증 평균: 0.9133 
# CategoricalNB의 정확도 :0.933 검증 평균: 0.9333
# ComplementNB의 정확도 :0.8 검증 평균: 0.6667
# DecisionTreeClassifier의 정확도 :0.967 검증 평균: 0.9467
# DummyClassifier의 정확도 :0.2 검증 평균: 0.2933 
# ExtraTreeClassifier의 정확도 :0.9 검증 평균: 0.9267
# ExtraTreesClassifier의 정확도 :0.967 검증 평균: 0.9533 
# GaussianNB의 정확도 :0.967 검증 평균: 0.9467
# GaussianProcessClassifier의 정확도 :0.967 검증 평균: 0.96 
# GradientBoostingClassifier의 정확도 :0.967 검증 평균: 0.96 
# HistGradientBoostingClassifier의 정확도 :0.967 검증 평균: 0.94 
# KNeighborsClassifier의 정확도 :1.0 검증 평균: 0.96
# LabelPropagation의 정확도 :1.0 검증 평균: 0.96 
# LabelSpreading의 정확도 :1.0 검증 평균: 0.96
# LinearDiscriminantAnalysis의 정확도 :1.0 검증 평균: 0.98 
# LinearSVC의 정확도 :1.0 검증 평균: 0.9667 
# LogisticRegression의 정확도 :0.967 검증 평균: 0.9667 
# LogisticRegressionCV의 정확도 :1.0 검증 평균: 0.9733 
# MLPClassifier의 정확도 :1.0 검증 평균: 0.9733 
# MultinomialNB의 정확도 :0.6 검증 평균: 0.9667
# NearestCentroid의 정확도 :0.967 검증 평균: 0.9333
# NuSVC의 정확도 :1.0 검증 평균: 0.9733 
# PassiveAggressiveClassifier의 정확도 :0.567 검증 평균: 0.7733 
# Perceptron의 정확도 :1.0 검증 평균: 0.78 
# QuadraticDiscriminantAnalysis의 정확도 :1.0 검증 평균: 0.98
# RadiusNeighborsClassifier의 정확도 :0.933 검증 평균: 0.9533 
# RandomForestClassifier의 정확도 :0.967 검증 평균: 0.96 
# RidgeClassifier의 정확도 :0.867 검증 평균: 0.84
# RidgeClassifierCV의 정확도 :0.867 검증 평균: 0.84
# SGDClassifier의 정확도 :0.567 검증 평균: 0.7867 
# SVC의 정확도 :1.0 검증 평균: 0.9667 