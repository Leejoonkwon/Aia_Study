
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
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.949, shuffle = True, random_state = 100
 )
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(test_set)
# print(y)
# print(y.shape) # (10886,)
print(x_train) #(10330, 8)
print(x_test) #(556, 8)
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
        
        
# ARDRegression 의  r2_score : 0.3026184734912225
# AdaBoostRegressor 의  r2_score : 0.14815877661367305
# BaggingRegressor 의  r2_score : 0.21655145604010229
# BayesianRidge 의  r2_score : 0.30230596997433956
# CCA 의  r2_score : -0.12661635845916908
# DecisionTreeRegressor 의  r2_score : -0.18738909661610248
# DummyRegressor 의  r2_score : -0.0228090722719978
# ElasticNet 의  r2_score : 0.22435082613923263
# ElasticNetCV 의  r2_score : 0.29639752105931483
# ExtraTreeRegressor 의  r2_score : -0.14410304251070238
# ExtraTreesRegressor 의  r2_score : 0.18254521362831444
# GammaRegressor 의  r2_score : 0.1794670411148085
# GaussianProcessRegressor 의  r2_score : -9607.719535446331
# GradientBoostingRegressor 의  r2_score : 0.35596679788154484
# HistGradientBoostingRegressor 의  r2_score : 0.3592232845592611
# HuberRegressor 의  r2_score : 0.3031558366056173
# IsotonicRegression 은 안나온 놈!!!
# KNeighborsRegressor 의  r2_score : 0.3026128470203169
# KernelRidge 의  r2_score : -0.23457065051147286
# Lars 의  r2_score : 0.302609058144347
# LarsCV 의  r2_score : 0.3023236497481119
# Lasso 의  r2_score : 0.30083626316462986
# LassoCV 의  r2_score : 0.30195302537837365
# LassoLars 의  r2_score : -0.0228090722719978
# LassoLarsCV 의  r2_score : 0.3023236497481119
# LassoLarsIC 의  r2_score : 0.3023778371790953
# LinearRegression 의  r2_score : 0.302609058144347
# LinearSVR 의  r2_score : 0.28735196389427087
# MLPRegressor 의  r2_score : 0.33699995956183504
# MultiOutputRegressor 은 안나온 놈!!!
# MultiTaskElasticNet 은 안나온 놈!!!
# MultiTaskElasticNetCV 은 안나온 놈!!!
# MultiTaskLasso 은 안나온 놈!!!
# MultiTaskLassoCV 은 안나온 놈!!!
# NuSVR 의  r2_score : 0.3014095739381424
# OrthogonalMatchingPursuit 의  r2_score : 0.14897358971003505
# OrthogonalMatchingPursuitCV 의  r2_score : 0.29785835509799374
# PLSCanonical 의  r2_score : -0.625223228110755
# PLSRegression 의  r2_score : 0.29664451077842713
# PassiveAggressiveRegressor 의  r2_score : 0.27147443723176234
# PoissonRegressor 의  r2_score : 0.28077335645588575
# RANSACRegressor 의  r2_score : 0.28120444917634935
# RadiusNeighborsRegressor 의  r2_score : 0.3592560922244765
# RandomForestRegressor 의  r2_score : 0.27234868152860103
# RegressorChain 은 안나온 놈!!!
# Ridge 의  r2_score : 0.30257945165318956
# RidgeCV 의  r2_score : 0.3023399757789438
# SGDRegressor 의  r2_score : 0.30143195591375105
# SVR 의  r2_score : 0.31384203982231873
# StackingRegressor 은 안나온 놈!!!
# TheilSenRegressor 의  r2_score : 0.30710155082870194
# TransformedTargetRegressor 의  r2_score : 0.302609058144347
# TweedieRegressor 의  r2_score : 0.17245469003608527
# VotingRegressor 은 안나온 놈!!!        