
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
path = 'D:\study_data\_data\_csv\kaggle_bike/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
print(train_set)

print(train_set.columns)


test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)

sampleSubmission = pd.read_csv(path + 'sampleSubmission.csv',#예측에서 쓸거야!!
                       index_col=0)
            
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
                                     workingday_out_index,
                                     weather_out_index,
                                     temp_out_index,
                                     atemp_out_index,
                                     humidity_out_index,
                                     windspeed_out_index,
                                     casual_out_index,registered_out_index),axis=None)
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

y = train_set_clean['count']

from sklearn.model_selection import KFold,cross_val_score,cross_val_predict

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,KFold,StratifiedKFold
from xgboost import XGBClassifier,XGBRegressor

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.8
       ,random_state=1234,shuffle=True)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits,shuffle=True,random_state=123)
# parameters = {'n_estimators':[100,200,300,400,500,1000], # 디폴트 100/ 1~inf 무한대 
# eta[기본값=0.3, 별칭: learning_rate] learning_rate':[0.1,0.2,0.3,0.4,0.5,0.7,1]
# max_depth': [None,3,4,5,6,7][기본값=6]
# gamma[기본값=0, 별칭: min_split_loss] [0,0.1,0.3,0.5,0.7,0.8,0.9,1]
# min_child_weight[기본값=1] 0~inf
# subsample[기본값=1][0,0.1,0.3,0.5,0.7,1] 0~1
# colsample_bytree [0,0.1,0.2,0.3,0.5,0.7,1]    [기본값=1] 0~1
# colsample_bylevel': [0,0.1,0.2,0.3,0.5,0.7,1] [기본값=1] 0~1
# 'colsample_bynode': [0,0.1,0.2,0.3,0.5,0.7,1] [기본값=1] 0~1
# 'reg_alpha' : [0,0.1 ,0.01, 0.001, 1 ,2 ,10]  [기본값=0] 0~inf /L1 절댓값 가중치 규제 
# 'reg_lambda' : [0,0.1 ,0.01, 0.001, 1 ,2 ,10]  [기본값=1] 0~inf /L2 절댓값 가중치 규제 
# max_delta_step[기본값=0]

parameters = {'n_estimators':[100],
              'learning_rate':[0.1],
              'max_depth': [None,3,4,5,6,7],
            #   'gamma' : [1],
            #   'min_child_weight' : [1],
            #   'subsample' : [1],
            #   'colsample_bytree' : [0.5],
            #   'colsample_bylevel': [1],
            #   'colsample_bynode': [1],
            #   'alpha' : [0],
            #   'lambda' : [0]
              } # 디폴트 6 


#2. 모델 
xgb = XGBRegressor(random_state=123,
                   )

model = GridSearchCV(xgb,parameters,cv=kfold,n_jobs=-1)
import time
start_time= time.time()
model.fit(x_train,y_train)
end_time= time.time()-start_time
# model.score(x_test,y_test)
result = model.score(x_test,y_test)
print('최적의 매개변수 : ',model.best_params_)
print('최상의 점수 : ',model.best_score_)
print('model.score :',result)
print('걸린 시간 : ',end_time
      )
###################################################
# 최적의 매개변수 :  {'n_estimators': 100}
# 최상의 점수 :  0.2985623764677555
# model.score : 0.3500904794006279
# 걸린 시간 :  9.94803237915039
###################################################
# 최적의 매개변수 :  {'learning_rate': 0.1, 'n_estimators': 100}
# 최상의 점수 :  0.3406808756484905
# model.score : 0.36434781056156407
# 걸린 시간 :  4.5136213302612305