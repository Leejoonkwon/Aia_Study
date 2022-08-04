import numpy as np
from sklearn.datasets import load_wine
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
#1.데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']
# print(x.shape, y.shape) #(178, 13) (178,)

# print(datasets.DESCR)
# print(datasets.feature_names)
x = datasets.data
y = datasets['target']
from sklearn.svm import LinearSVC 
from sklearn.svm import LinearSVR 



print(x.shape, y.shape) #178, 13) (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.15,shuffle=True ,random_state=100)
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
# scaler.transform(x_train)
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
# AdaBoostClassifier 의 정답률 : 0.8148148148148148
# BaggingClassifier 의 정답률 : 0.8888888888888888  
# BernoulliNB 의 정답률 : 0.4074074074074074
# CalibratedClassifierCV 의 정답률 : 1.0
# CategoricalNB 은 안나온 놈!!!
# ClassifierChain 은 안나온 놈!!!
# ComplementNB 의 정답률 : 0.8518518518518519       
# DecisionTreeClassifier 의 정답률 : 0.9259259259259259
# DummyClassifier 의 정답률 : 0.4074074074074074    
# ExtraTreeClassifier 의 정답률 : 0.8518518518518519
# ExtraTreesClassifier 의 정답률 : 1.0
# GaussianNB 의 정답률 : 1.0
# GaussianProcessClassifier 의 정답률 : 0.9629629629629629
# GradientBoostingClassifier 의 정답률 : 0.9629629629629629
# HistGradientBoostingClassifier 의 정답률 : 0.9259259259259259
# KNeighborsClassifier 의 정답률 : 0.8888888888888888
# LabelPropagation 의 정답률 : 0.9259259259259259   
# LabelSpreading 의 정답률 : 0.9259259259259259     
# LinearDiscriminantAnalysis 의 정답률 : 0.9629629629629629
# LinearSVC 의 정답률 : 0.9629629629629629
# LogisticRegression 의 정답률 : 1.0
# LogisticRegressionCV 의 정답률 : 0.9629629629629629
# MLPClassifier 의 정답률 : 1.0
# MultiOutputClassifier 은 안나온 놈!!!
# MultinomialNB 의 정답률 : 0.9629629629629629      
# NearestCentroid 의 정답률 : 0.9259259259259259    
# NuSVC 의 정답률 : 0.9259259259259259
# OneVsOneClassifier 은 안나온 놈!!!
# OneVsRestClassifier 은 안나온 놈!!!
# OutputCodeClassifier 은 안나온 놈!!!
# PassiveAggressiveClassifier 의 정답률 : 0.9259259259259259
# Perceptron 의 정답률 : 0.9259259259259259
# QuadraticDiscriminantAnalysis 의 정답률 : 1.0
# RadiusNeighborsClassifier 의 정답률 : 0.8888888888888888
# RandomForestClassifier 의 정답률 : 1.0
# RidgeClassifier 의 정답률 : 1.0
# RidgeClassifierCV 의 정답률 : 0.9629629629629629  
# SGDClassifier 의 정답률 : 0.9629629629629629      
# SVC 의 정답률 : 0.9629629629629629
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!        