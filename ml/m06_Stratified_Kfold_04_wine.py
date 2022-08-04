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
# AdaBoostClassifier의 정확도 :0.833 검증 평균: 0.8192 
# BaggingClassifier의 정확도 :0.917 검증 평균: 0.9721 
# BernoulliNB의 정확도 :0.889 검증 평균: 0.399
# CalibratedClassifierCV의 정확도 :0.944 검증 평균: 0.927 
# DecisionTreeClassifier의 정확도 :0.833 검증 평균: 0.9159 
# DummyClassifier의 정확도 :0.417 검증 평균: 0.399
# ExtraTreeClassifier의 정확도 :0.861 검증 평균: 0.8432 
# ExtraTreesClassifier의 정확도 :1.0 검증 평균: 0.9832 
# GaussianNB의 정확도 :1.0 검증 평균: 0.9662
# GaussianProcessClassifier의 정확도 :0.944 검증 평균: 0.4833
# GradientBoostingClassifier의 정확도 :0.889 검증 평균: 0.9551
# HistGradientBoostingClassifier의 정확도 :0.889 검증 평균: 0.9665
# KNeighborsClassifier의 정확도 :0.917 검증 평균: 0.6808 
# LabelPropagation의 정확도 :0.917 검증 평균: 0.5 
# LabelSpreading의 정확도 :0.917 검증 평균: 0.5 
# LinearDiscriminantAnalysis의 정확도 :0.972 검증 평균: 0.9889
# LinearSVC의 정확도 :0.972 검증 평균: 0.8206 
# LogisticRegression의 정확도 :0.944 검증 평균: 0.9495 
# LogisticRegressionCV의 정확도 :0.972 검증 평균: 0.9663 
# MLPClassifier의 정확도 :0.944 검증 평균: 0.4389 
# NearestCentroid의 정확도 :0.944 검증 평균: 0.7306        
# NuSVC의 정확도 :0.972 검증 평균: 0.8879
# PassiveAggressiveClassifier의 정확도 :0.944 검증 평균: 0.5898
# Perceptron의 정확도 :0.944 검증 평균: 0.5613 
# QuadraticDiscriminantAnalysis의 정확도 :1.0 검증 평균: 0.9944
# RandomForestClassifier의 정확도 :1.0 검증 평균: 0.9832 
# RidgeClassifier의 정확도 :0.972 검증 평균: 0.9887        
# RidgeClassifierCV의 정확도 :0.972 검증 평균: 0.9832 
# SGDClassifier의 정확도 :0.917 검증 평균: 0.6129 
# SVC의 정확도 :0.972 검증 평균: 0.6911 