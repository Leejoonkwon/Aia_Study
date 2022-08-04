import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import warnings 
warnings.filterwarnings('ignore')
#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25,
                                                    shuffle=True ,random_state=100)
print(y_train,y_test)
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler() 
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test)

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
        print(name,'의 정답률 :',acc)
    except:
        #continue
        print(name,'은 안나온 놈!!!')
# AdaBoostClassifier 의 정답률 : 0.9473684210526315
# BaggingClassifier 의 정답률 : 0.9473684210526315
# BernoulliNB 의 정답률 : 0.3157894736842105        
# CalibratedClassifierCV 의 정답률 : 0.9210526315789473
# CategoricalNB 은 안나온 놈!!!
# ClassifierChain 은 안나온 놈!!!
# ComplementNB 의 정답률 : 0.7368421052631579       
# DecisionTreeClassifier 의 정답률 : 0.9473684210526315
# DummyClassifier 의 정답률 : 0.2631578947368421    
# ExtraTreeClassifier 의 정답률 : 0.9210526315789473
# ExtraTreesClassifier 의 정답률 : 0.9473684210526315
# GaussianNB 의 정답률 : 0.9473684210526315
# GaussianProcessClassifier 의 정답률 : 0.9473684210526315
# GradientBoostingClassifier 의 정답률 : 0.9473684210526315
# HistGradientBoostingClassifier 의 정답률 : 0.9473684210526315
# KNeighborsClassifier 의 정답률 : 0.9736842105263158
# LabelPropagation 의 정답률 : 0.9473684210526315   
# LabelSpreading 의 정답률 : 0.9473684210526315     
# LinearDiscriminantAnalysis 의 정답률 : 1.0        
# LinearSVC 의 정답률 : 0.9473684210526315
# LogisticRegression 의 정답률 : 0.9473684210526315 
# LogisticRegressionCV 의 정답률 : 1.0
# MLPClassifier 의 정답률 : 0.9473684210526315
# MultiOutputClassifier 은 안나온 놈!!!
# MultinomialNB 의 정답률 : 0.631578947368421       
# NearestCentroid 의 정답률 : 0.9473684210526315    
# NuSVC 의 정답률 : 0.9473684210526315
# OneVsOneClassifier 은 안나온 놈!!!
# OneVsRestClassifier 은 안나온 놈!!!
# OutputCodeClassifier 은 안나온 놈!!!
# PassiveAggressiveClassifier 의 정답률 : 0.9210526315789473
# Perceptron 의 정답률 : 0.9473684210526315
# QuadraticDiscriminantAnalysis 의 정답률 : 1.0     
# RadiusNeighborsClassifier 의 정답률 : 0.3684210526315789
# RandomForestClassifier 의 정답률 : 0.9473684210526315
# RidgeClassifier 의 정답률 : 0.8947368421052632    
# RidgeClassifierCV 의 정답률 : 0.8421052631578947  
# SGDClassifier 의 정답률 : 0.9210526315789473      
# SVC 의 정답률 : 0.9736842105263158
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!  