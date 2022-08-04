
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC 
from sklearn.svm import LinearSVR 
from sklearn.metrics import r2_score


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data #데이터를 리스트 형태로 불러올 때 함
y = datasets.target
x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.8,shuffle=True,random_state=100)
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
# print(x.shape, y.shape) #(506, 13)-> 13개의 피쳐 (506,) 
print(x_train.shape) #(16512, 8)
print(x_test.shape) #(16512, 8)

#2. 모델 구성
import warnings 
warnings.filterwarnings('ignore')
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
# ARDRegression 의  r2_score : 0.6224383871562595
# AdaBoostRegressor 의  r2_score : 0.3006709758445939
# BaggingRegressor 의  r2_score : 0.8014298564941571
# BayesianRidge 의  r2_score : 0.6223160905876419
# CCA 의  r2_score : 0.5734820077747389
# DecisionTreeRegressor 의  r2_score : 0.6325465394681307
# DummyRegressor 의  r2_score : -0.0001669012609140097
# ElasticNet 의  r2_score : 0.20655232918577715
# ElasticNetCV 의  r2_score : 0.6221800004919685
# ExtraTreeRegressor 의  r2_score : 0.640880286505703
# ExtraTreesRegressor 의  r2_score : 0.8242432669828847
# GammaRegressor 의  r2_score : 0.39082855313962095
# GaussianProcessRegressor 의  r2_score : -7311.745047405698
# GradientBoostingRegressor 의  r2_score : 0.7982915560729515
# HistGradientBoostingRegressor 의  r2_score : 0.845626866138834
# HuberRegressor 의  r2_score : 0.6077517759907334
# IsotonicRegression 은 안나온 놈!!!
# KNeighborsRegressor 의  r2_score : 0.6937192268681903
# KernelRidge 의  r2_score : -2.5185626906188934
# Lars 의  r2_score : 0.6223138107295286
# LarsCV 의  r2_score : 0.6223138107295286
# Lasso 의  r2_score : -0.0001669012609140097
# LassoCV 의  r2_score : 0.6221482727146265
# LassoLars 의  r2_score : -0.0001669012609140097
# LassoLarsCV 의  r2_score : 0.6223138107295286
# LassoLarsIC 의  r2_score : 0.6223138107295286
# LinearRegression 의  r2_score : 0.6223138107295286
# LinearSVR 의  r2_score : 0.5816617205854815
# MLPRegressor 의  r2_score : 0.79095717785086
# MultiOutputRegressor 은 안나온 놈!!!
# MultiTaskElasticNet 은 안나온 놈!!!
# MultiTaskElasticNetCV 은 안나온 놈!!!
# MultiTaskLasso 은 안나온 놈!!!
# MultiTaskLassoCV 은 안나온 놈!!!
# NuSVR 의  r2_score : 0.7533124924810992
# OrthogonalMatchingPursuit 의  r2_score : 0.49258838196670585
# OrthogonalMatchingPursuitCV 의  r2_score : 0.6153466329492191
# PLSCanonical 의  r2_score : 0.4132782496009729
# PLSRegression 의  r2_score : 0.5478098077273479
# PassiveAggressiveRegressor 의  r2_score : -1.8819124513189274
# PoissonRegressor 의  r2_score : 0.4578498155081445
# RANSACRegressor 의  r2_score : 0.03977291240334535
# RadiusNeighborsRegressor 은 안나온 놈!!!
# RandomForestRegressor 의  r2_score : 0.8186214381269212
# RegressorChain 은 안나온 놈!!!
# Ridge 의  r2_score : 0.6223151803195337
# RidgeCV 의  r2_score : 0.6223212296757796
# SGDRegressor 의  r2_score : 0.6191663473661351
# SVR 의  r2_score : 0.7525300692785053
# StackingRegressor 은 안나온 놈!!!
# TheilSenRegressor 의  r2_score : -0.6064447825769825
# TransformedTargetRegressor 의  r2_score : 0.6223138107295286
# TweedieRegressor 의  r2_score : 0.4029164271580993
# VotingRegressor 은 안나온 놈!!!        