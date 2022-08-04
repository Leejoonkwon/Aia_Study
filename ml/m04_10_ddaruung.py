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
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.919, shuffle = True, random_state = 100)
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(y)
#2. 모델 구성
import warnings 
warnings.filterwarnings('ignore')
from sklearn.utils import all_estimators
from sklearn.metrics import r2_score

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
        
# ARDRegression 의  r2_score : 0.6056567655491609
# AdaBoostRegressor 의  r2_score : 0.6079428189033573
# BaggingRegressor 의  r2_score : 0.7938740886224713
# BayesianRidge 의  r2_score : 0.606324848209921
# CCA 의  r2_score : 0.34271777593613184
# DecisionTreeRegressor 의  r2_score : 0.6874248117030459
# DummyRegressor 의  r2_score : -0.0009738266862373557
# ElasticNet 의  r2_score : 0.21292888956672262
# ElasticNetCV 의  r2_score : 0.5819046279251654
# ExtraTreeRegressor 의  r2_score : 0.5052559781928601
# ExtraTreesRegressor 의  r2_score : 0.8096052970978216
# GammaRegressor 의  r2_score : 0.127319349282783
# GaussianProcessRegressor 의  r2_score : -122.54933625316346
# GradientBoostingRegressor 의  r2_score : 0.795989847398077
# HistGradientBoostingRegressor 의  r2_score : 0.8451695969086769
# HuberRegressor 의  r2_score : 0.592917765778473
# IsotonicRegression 은 안나온 놈!!!
# KNeighborsRegressor 의  r2_score : 0.7528726805652388
# KernelRidge 의  r2_score : 0.60529038612709
# Lars 의  r2_score : 0.6069571332489507
# LarsCV 의  r2_score : 0.6069571332489507
# Lasso 의  r2_score : 0.581082440001349
# LassoCV 의  r2_score : 0.606737916119416
# LassoLars 의  r2_score : 0.2666333448807906
# LassoLarsCV 의  r2_score : 0.6069571332489507
# LassoLarsIC 의  r2_score : 0.6056827279474601
# LinearRegression 의  r2_score : 0.6069571332489507
# LinearSVR 의  r2_score : 0.48991962993979876
# MLPRegressor 의  r2_score : 0.5527168000889462
# MultiOutputRegressor 은 안나온 놈!!!
# MultiTaskElasticNet 은 안나온 놈!!!
# MultiTaskElasticNetCV 은 안나온 놈!!!
# MultiTaskLasso 은 안나온 놈!!!
# MultiTaskLassoCV 은 안나온 놈!!!
# NuSVR 의  r2_score : 0.514980126930273
# OrthogonalMatchingPursuit 의  r2_score : 0.37374641842098966
# OrthogonalMatchingPursuitCV 의  r2_score : 0.5889731563427768
# PLSCanonical 의  r2_score : -0.19726599081937723
# PLSRegression 의  r2_score : 0.5896076600493537
# PassiveAggressiveRegressor 의  r2_score : 0.5786436255036418
# PoissonRegressor 의  r2_score : 0.6310670411391921
# RANSACRegressor 의  r2_score : 0.4541564563998337
# RadiusNeighborsRegressor 의  r2_score : 0.28172929524011037
# RandomForestRegressor 의  r2_score : 0.8096354578261988
# RegressorChain 은 안나온 놈!!!
# Ridge 의  r2_score : 0.6058082211081597
# RidgeCV 의  r2_score : 0.6058082211081732
# SGDRegressor 의  r2_score : 0.5981031209774927
# SVR 의  r2_score : 0.5035232735894759
# StackingRegressor 은 안나온 놈!!!
# TheilSenRegressor 의  r2_score : 0.5831361082311103
# TransformedTargetRegressor 의  r2_score : 0.6069571332489507
# TweedieRegressor 의  r2_score : 0.13286490886612146
# VotingRegressor 은 안나온 놈!!        