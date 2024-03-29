
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
from sklearn.metrics import r2_score
import numpy as np
#1. 데이터
path = 'C:\_data\kaggle_bike/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
print(train_set)

print(train_set.columns)


test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)

# sampleSubmission = pd.read_csv(path + 'sampleSubmission.csv',#예측에서 쓸거야!!
#                        index_col=0)
            
# print(test_set)
# print(test_set.shape) #(6493, 8) #train_set과 열 값이 '1'차이 나는 건 count를 제외했기 때문이다.예측 단계에서 값을 대입

# print(train_set.info()) #null은 누락된 값이라고 하고 "결측치"라고도 한다.
# print(train_set.describe()) 


###### 결측치 처리 1.중위 ##### 
print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
train_set = train_set.fillna(train_set.median())
print(train_set.isnull().sum())
print(train_set.shape)
test_set = test_set.fillna(test_set.median())

###### 결측치 처리 2.interpolate #####
# print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
# train_set = train_set.interpolate()
# print(train_set.isnull().sum())
# print(train_set.shape)
# test_set = test_set.interpolate()

####### 결측치 처리 3.mean #####
# print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
# train_set = train_set.fillna(train_set.mean())
# print(train_set.isnull().sum())
# print(train_set.shape)
# test_set = test_set.fillna(test_set.mean())

####### 결측치 처리 3.drop #####
# print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
# train_set = train_set.dropna()
# print(train_set.isnull().sum())
# print(train_set.shape)
# test_set = test_set.dropna()

def outliers(data_out):
    quartile_1, q2 , quartile_3 = np.percentile(data_out,
                                               [25,50,75]) # percentile 백분위
    print("1사분위 : ",quartile_1) # 25% 위치인수를 기점으로 사이에 값을 구함
    print("q2 : ",q2) # 50% median과 동일 
    print("3사분위 : ",quartile_3) # 75% 위치인수를 기점으로 사이에 값을 구함
    iqr =quartile_3-quartile_1  # 75% -25%
    print("iqr :" ,iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound)|
                    (data_out<lower_bound))

# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],
season_out_index= outliers(train_set['season'])[0]
holiday_out_index= outliers(train_set['holiday'])[0]
workingday_out_index= outliers(train_set['workingday'])[0]
weather_out_index= outliers(train_set['weather'])[0]
temp_out_index= outliers(train_set['temp'])[0]
atemp_out_index= outliers(train_set['atemp'])[0]
humidity_out_index= outliers(train_set['humidity'])[0]
windspeed_out_index= outliers(train_set['windspeed'])[0]
casual_out_index= outliers(train_set['casual'])[0]
registered_out_index= outliers(train_set['registered'])[0]
# print(train_set2.loc[season_out_index,'season'])
lead_outlier_index = np.concatenate((season_out_index,
                                     holiday_out_index,
                                    #  workingday_out_index,
                                     weather_out_index,
                                     temp_out_index,
                                    #  atemp_out_index,
                                    #  humidity_out_index,
                                     windspeed_out_index,
                                     casual_out_index,
                                     registered_out_index),axis=None)
print(len(lead_outlier_index)) #161개 
print(lead_outlier_index)
lead_not_outlier_index = []
for i in train_set.index:
    if i not in lead_outlier_index :
        lead_not_outlier_index.append(i)
train_set_clean = train_set.loc[lead_not_outlier_index]      
train_set_clean = train_set_clean.reset_index(drop=True)
print(train_set_clean)
x = train_set_clean.drop([ 'casual', 'registered','count'],axis=1) #axis는 컬럼 


# print(x.columns)
# print(x.shape) #(10886, 8)

y = train_set_clean['count']

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split,KFold 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import make_pipeline


x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8,random_state=1234,
)
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import RobustScaler,QuantileTransformer # 이상치에 강함
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

aaa =[StandardScaler(),MinMaxScaler(),RobustScaler(),
      QuantileTransformer(),PowerTransformer(method='yeo-johnson'),
    #   PowerTransformer(method='box-cox')
      ]
for i in aaa:
    scaler = i
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = RandomForestRegressor()
    model.fit(x_train,y_train)
    y_predict =model.predict(x_test)
    results = r2_score(y_test,y_predict)
    print("{0} : {1:4f} ".format(i,round(results, 4)))
######Bagging 후 r2 model xgb
# model.score : 0.37089630515782024

# 보팅 결과 : 0.3413
# CatBoostRegressor score : 0.335828
# XGBRegressor score : 0.309432
# LGBMRegressor score : 0.327279
# RandomForestRegressor score : 0.273691

# poly 스코어 : 0.3193545166426033
# 폴리 CV :  [0.32321845 0.33069049 0.25699658 0.29058442 0.29626553]
# 폴리 CV 엔빵 :  0.29955109531238777

# StandardScaler() : 0.326000 
# MinMaxScaler() : 0.327700 
# RobustScaler() : 0.324300 
# QuantileTransformer() : 0.331300 
# PowerTransformer() : 0.321600 



