
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
from sklearn.model_selection import KFold,cross_val_score,cross_val_predict
import warnings 
from sklearn.metrics import r2_score

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
import numpy as np

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
        # acc = accuracy_score(y_test,y_predict)
        r2  = r2_score(y_test,y_predict)
        scores = cross_val_score(model, x, y, cv=kfold)
        # print('{}의 정확도 {}의 ',name,'의 정답률 :',acc)
        print('{}의 r2_score :{} 검증 평균: {} '.format(name,round(r2,3),round(np.mean(scores),4)))
       
    except:
        continue 
    
# ARDRegression의 r2_score :0.862 검증 평균: 0.7811 
# AdaBoostRegressor의 r2_score :0.817 검증 평균: 0.7969 
# BaggingRegressor의 r2_score :0.851 검증 평균: 0.8462 
# BayesianRidge의 r2_score :0.865 검증 평균: 0.7863 
# CCA의 r2_score :-0.395 검증 평균: -0.6554 
# DecisionTreeRegressor의 r2_score :0.76 검증 평균: 0.7169 
# DummyRegressor의 r2_score :-0.0 검증 평균: -0.0026
# ElasticNet의 r2_score :0.855 검증 평균: 0.7844 
# ElasticNetCV의 r2_score :0.202 검증 평균: 0.5837 
# ExtraTreeRegressor의 r2_score :0.708 검증 평균: 0.6639 
# ExtraTreesRegressor의 r2_score :0.887 검증 평균: 0.8645 
# GammaRegressor의 r2_score :0.868 검증 평균: -0.0029 
# GaussianProcessRegressor의 r2_score :-5.068 검증 평균: -5.2023 
# GradientBoostingRegressor의 r2_score :0.902 검증 평균: 0.8543 
# HistGradientBoostingRegressor의 r2_score :0.888 검증 평균: 0.864 
# HuberRegressor의 r2_score :0.875 검증 평균: 0.6797 
# KNeighborsRegressor의 r2_score :0.764 검증 평균: 0.6481 
# KernelRidge의 r2_score :-4.433 검증 평균: 0.7777 
# Lars의 r2_score :0.57 검증 평균: 0.3621 
# LarsCV의 r2_score :0.835 검증 평균: 0.7056 
# Lasso의 r2_score :0.863 검증 평균: 0.7778 
# LassoCV의 r2_score :0.865 검증 평균: 0.6824 
# LassoLars의 r2_score :0.864 검증 평균: 0.7788 
# LassoLarsCV의 r2_score :0.864 검증 평균: 0.7861 
# LassoLarsIC의 r2_score :0.861 검증 평균: 0.7873 
# LinearRegression의 r2_score :0.863 검증 평균: 0.7778 
# LinearSVR의 r2_score :-5.093 검증 평균: 0.0985 
# MLPRegressor의 r2_score :-4.97 검증 평균: 0.6264 
# NuSVR의 r2_score :-0.015 검증 평균: -0.0146 
# OrthogonalMatchingPursuit의 r2_score :0.858 검증 평균: 0.7697 
# OrthogonalMatchingPursuitCV의 r2_score :0.858 검증 평균: 0.7697 
# PLSCanonical의 r2_score :-4.071 검증 평균: -4.761 
# PLSRegression의 r2_score :0.862 검증 평균: 0.7967 
# PassiveAggressiveRegressor의 r2_score :0.874 검증 평균: 0.3959 
# PoissonRegressor의 r2_score :0.892 검증 평균: -0.0029 
# RANSACRegressor의 r2_score :0.812 검증 평균: 0.7411 
# RadiusNeighborsRegressor의 r2_score :-1.3186819879354341e+28 검증 평균: -5.489054505378216e+27
# RandomForestRegressor의 r2_score :0.875 검증 평균: 0.8504 
# Ridge의 r2_score :0.863 검증 평균: 0.7782 
# RidgeCV의 r2_score :0.864 검증 평균: 0.7789 
# SGDRegressor의 r2_score :0.855 검증 평균: -4.6999246451209596e+23 
# SVR의 r2_score :-0.067 검증 평균: -0.0526 
# TheilSenRegressor의 r2_score :0.753 검증 평균: 0.7188 
# TransformedTargetRegressor의 r2_score :0.863 검증 평균: 0.7778 
# TweedieRegressor의 r2_score :0.838 검증 평균: 0.6598     
    
    