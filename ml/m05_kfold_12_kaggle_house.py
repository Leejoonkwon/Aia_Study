
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import null
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten,LSTM,Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm_notebook
from sklearn.svm import LinearSVC 
from sklearn.svm import LinearSVR 
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
#1. 데이터
path = 'D:\study_data\_data\_csv\kaggle_house/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)
drop_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
test_set.drop(drop_cols, axis = 1, inplace =True)
# submission = pd.read_csv(path + 'submission.csv',#예측에서 쓸거야!!
#                        index_col=0)
print(train_set)

print(train_set.shape) #(1459, 10)

train_set.drop(drop_cols, axis = 1, inplace =True)
cols = ['MSZoning', 'Street','LandContour','Neighborhood','Condition1','Condition2',
                'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation',
                'Heating','GarageType','SaleType','SaleCondition','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                'BsmtFinType2','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',
                'FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive','LotShape',
                'Utilities','LandSlope','BldgType','HouseStyle','LotConfig']

for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])


print(test_set)
print(train_set.shape) #(1460,76)
print(test_set.shape) #(1459, 75) #train_set과 열 값이 '1'차이 나는 건 count를 제외했기 때문이다.예측 단계에서 값을 대입

print(train_set.columns)
print(train_set.info()) #null은 누락된 값이라고 하고 "결측치"라고도 한다.
print(train_set.describe()) 

###### 결측치 처리 1.제거##### dropna 사용
print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
train_set = train_set.fillna(train_set.median())
print(train_set.isnull().sum())
print(train_set.shape)
test_set = test_set.fillna(test_set.median())

x = train_set.drop(['SalePrice'],axis=1) #axis는 컬럼 
print(x.columns)
print(x.shape) #(1460, 75)

y = train_set['SalePrice']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.89, shuffle = True, random_state = 68
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
print(y)
print(y.shape) # (1460,)
print(x_train.shape) #(1299, 75)
print(x_test.shape) #(161, 75)
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
# ARDRegression 의  r2_score : 0.8416325093866162
# AdaBoostRegressor 의  r2_score : 0.7861545626167085
# BaggingRegressor 의  r2_score : 0.8623032218521639
# BayesianRidge 의  r2_score : 0.8427415994475478
# CCA 의  r2_score : -0.5370589336084155
# DecisionTreeRegressor 의  r2_score : 0.7219190061148253
# DummyRegressor 의  r2_score : -0.0026988215215173472
# ElasticNet 의  r2_score : 0.8329674095353258
# ElasticNetCV 의  r2_score : 0.02513393879826553
# ExtraTreeRegressor 의  r2_score : 0.6906149023558702
# ExtraTreesRegressor 의  r2_score : 0.8511374520228919
# GammaRegressor 의  r2_score : -0.002698821521516903
# GaussianProcessRegressor 의  r2_score : -4.936672015120445
# GradientBoostingRegressor 의  r2_score : 0.9103411548982281
# HistGradientBoostingRegressor 의  r2_score : 0.8813246335172752
# HuberRegressor 의  r2_score : 0.6636042104463955
# IsotonicRegression 은 안나온 놈!!!
# KNeighborsRegressor 의  r2_score : 0.6934792699382527
# KernelRidge 의  r2_score : 0.8398034301698007
# Lars 의  r2_score : 0.735889198667446
# LarsCV 의  r2_score : 0.8134304538129123
# Lasso 의  r2_score : 0.8346015923940853
# LassoCV 의  r2_score : 0.8396613180067448
# LassoLars 의  r2_score : 0.8353262442100537
# LassoLarsCV 의  r2_score : 0.8306039311952513
# LassoLarsIC 의  r2_score : 0.8291553629584953
# LinearRegression 의  r2_score : 0.8345040236882806
# LinearSVR 의  r2_score : -3.640775043465629
# MLPRegressor 의  r2_score : -4.0850148205494365
# MultiOutputRegressor 은 안나온 놈!!!
# MultiTaskElasticNet 은 안나온 놈!!!
# MultiTaskElasticNetCV 은 안나온 놈!!!
# MultiTaskLasso 은 안나온 놈!!!
# MultiTaskLassoCV 은 안나온 놈!!!
# NuSVR 의  r2_score : -0.003930958682101915
# OrthogonalMatchingPursuit 의  r2_score : 0.8130542585974221
# OrthogonalMatchingPursuitCV 의  r2_score : 0.8130542585974221
# PLSCanonical 의  r2_score : -5.073290941041052
# PLSRegression 의  r2_score : 0.8400653670006811
# PassiveAggressiveRegressor 의  r2_score : 0.6372960781679096
# PoissonRegressor 의  r2_score : -0.002698821521516903
# RANSACRegressor 의  r2_score : 0.7852266390867759
# RadiusNeighborsRegressor 의  r2_score : -5.217229750183084
# RandomForestRegressor 의  r2_score : 0.8954173773535534
# RegressorChain 은 안나온 놈!!!
# Ridge 의  r2_score : 0.8363354686689815
# RidgeCV 의  r2_score : 0.8400411070836573
# SGDRegressor 의  r2_score : -5.7211332746562534e+17
# SVR 의  r2_score : -0.031732730483525096
# StackingRegressor 은 안나온 놈!!!        