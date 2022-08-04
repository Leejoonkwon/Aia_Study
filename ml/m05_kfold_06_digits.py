from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import r2_score,accuracy_score
import numpy as np
#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.95,shuffle=True, random_state=12 ) 
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
# AdaBoostClassifier의 정확도 :0.278 검증 평균: 0.2738 
# BaggingClassifier의 정확도 :0.922 검증 평균: 0.9315 
# BernoulliNB의 정확도 :0.897 검증 평균: 0.8514 
# CalibratedClassifierCV의 정확도 :0.953 검증 평균: 0.9622 
# DecisionTreeClassifier의 정확도 :0.878 검증 평균: 0.8492 
# DummyClassifier의 정확도 :0.089 검증 평균: 0.0785
# ExtraTreeClassifier의 정확도 :0.767 검증 평균: 0.7863 
# ExtraTreesClassifier의 정확도 :0.978 검증 평균: 0.9833 
# GaussianNB의 정확도 :0.797 검증 평균: 0.8392 
# GaussianProcessClassifier의 정확도 :0.983 검증 평균: 0.1046 
# GradientBoostingClassifier의 정확도 :0.967 검증 평균: 0.9649 
# HistGradientBoostingClassifier의 정확도 :0.972 검증 평균: 0.9727 
# KNeighborsClassifier의 정확도 :0.978 검증 평균: 0.9883 
# LabelPropagation의 정확도 :0.975 검증 평균: 0.1002 
# LabelSpreading의 정확도 :0.975 검증 평균: 0.1002 
# LinearDiscriminantAnalysis의 정확도 :0.944 검증 평균: 0.9533 
# LinearSVC의 정확도 :0.964 검증 평균: 0.9516 
# LogisticRegression의 정확도 :0.981 검증 평균: 0.9716 
# LogisticRegressionCV의 정확도 :0.978 검증 평균: 0.9705 
# MLPClassifier의 정확도 :0.989 검증 평균: 0.98 
# NearestCentroid의 정확도 :0.889 검증 평균: 0.8998
# NuSVC의 정확도 :0.961 검증 평균: 0.9616 
# PassiveAggressiveClassifier의 정확도 :0.939 검증 평균: 0.9549 
# Perceptron의 정확도 :0.928 검증 평균: 0.9466 
# QuadraticDiscriminantAnalysis의 정확도 :0.85 검증 평균: 0.8592 
# RandomForestClassifier의 정확도 :0.969 검증 평균: 0.975 
# RidgeClassifier의 정확도 :0.939 검증 평균: 0.9338 
# RidgeClassifierCV의 정확도 :0.939 검증 평균: 0.9343 
# SGDClassifier의 정확도 :0.95 검증 평균: 0.9499 
# SVC의 정확도 :0.989 검증 평균: 0.9883     