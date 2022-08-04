#2. 모델 구성
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import accuracy_score,r2_score
# matplotlib.rcParams['font.family']='Malgun Gothic'
# matplotlib.rcParams['axes.unicode_minus']=False
import time
from sklearn.svm import LinearSVC

#1. 데이터
datasets = load_boston()
x = datasets.data #데이터를 리스트 형태로 불러올 때 함
y = datasets.target
x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.75,shuffle=True,random_state=100)
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
        r2 = r2_score(y_test,y_predict)
        print(name,'의  r2_score :',r2)
    except:
        #continue
        print(name,'은 안나온 놈!!!')
# ARDRegression 의  r2_score : 0.724629784111766
# AdaBoostRegressor 의  r2_score : 0.8019894930966173
# BaggingRegressor 의  r2_score : 0.8683387566100997
# BayesianRidge 의  r2_score : 0.7050490451798533
# CCA 의  r2_score : 0.7338225823667008
# DecisionTreeRegressor 의  r2_score : 0.7519472398435694
# DummyRegressor 의  r2_score : -0.004482921853908417
# ElasticNet 의  r2_score : 0.6634747853513165
# ElasticNetCV 의  r2_score : 0.6496490851857444
# ExtraTreeRegressor 의  r2_score : 0.7032660238365722
# ExtraTreesRegressor 의  r2_score : 0.8688987900910083
# GammaRegressor 의  r2_score : -0.004482921853908417
# GaussianProcessRegressor 의  r2_score : -4.874295740024044
# GradientBoostingRegressor 의  r2_score : 0.9022802000348283
# HistGradientBoostingRegressor 의  r2_score : 0.8424718794338251
# IsotonicRegression 은 안나온 놈!!!
# KNeighborsRegressor 의  r2_score : 0.44371014889060933
# KernelRidge 의  r2_score : 0.6885712695985845
# Lars 의  r2_score : 0.7246154314616735
# LarsCV 의  r2_score : 0.719911028446091
# Lasso 의  r2_score : 0.6577184083896714
# LassoCV 의  r2_score : 0.6800107758222457
# LassoLars 의  r2_score : -0.004482921853908417
# LassoLarsCV 의  r2_score : 0.7234544783131645
# LassoLarsIC 의  r2_score : 0.7246154314616735
# LinearRegression 의  r2_score : 0.724615431461674
# LinearSVR 의  r2_score : 0.5914911809023595
# MLPRegressor 의  r2_score : 0.640886473540828
# MultiOutputRegressor 은 안나온 놈!!!
# MultiTaskElasticNet 은 안나온 놈!!!
# MultiTaskElasticNetCV 은 안나온 놈!!!
# MultiTaskLasso 은 안나온 놈!!!
# MultiTaskLassoCV 은 안나온 놈!!!
# NuSVR 의  r2_score : 0.2299874485909036
# OrthogonalMatchingPursuit 의  r2_score : 0.5464643025805889
# OrthogonalMatchingPursuitCV 의  r2_score : 0.6922355818381616
# PLSCanonical 의  r2_score : -1.7966326799971784
# PLSRegression 의  r2_score : 0.6907416924949574
# PassiveAggressiveRegressor 의  r2_score : -0.12784206545485644
# PoissonRegressor 의  r2_score : 0.7366551730989979
# RANSACRegressor 의  r2_score : -0.34566258565131824
# RadiusNeighborsRegressor 은 안나온 놈!!!
# RandomForestRegressor 의  r2_score : 0.8764484321082449
# RegressorChain 은 안나온 놈!!!
# Ridge 의  r2_score : 0.716467619700932
# RidgeCV 의  r2_score : 0.7234106470993683
# SGDRegressor 의  r2_score : -5.089994471785672e+26
# SVR 의  r2_score : 0.2060097280934967
# StackingRegressor 은 안나온 놈!!!
# TheilSenRegressor 의  r2_score : 0.6704835278611374
# TransformedTargetRegressor 의  r2_score : 0.724615431461674
# TweedieRegressor 의  r2_score : 0.6497023253424977
# VotingRegressor 은 안나온 놈!!!        