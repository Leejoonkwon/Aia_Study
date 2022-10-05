
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten,LSTM,Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
import pandas as pd
from sklearn.svm import LinearSVC 
from sklearn.svm import LinearSVR 
import numpy as np 
#1. 데이터
path = 'D:\study_data\_data\_csv\_ddarung/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        )
test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       )
submission = pd.read_csv(path + 'submission.csv',#예측에서 쓸거야!!
                       index_col=0)
                       
# print(test_set)
# print(test_set.shape) #(715, 9) #train_set과 열 값이 '1'차이 나는 건 count를 제외했기 때문이다.예측 단계에서 값을 대입

# print(test_set.columns)
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
# train_set2 = train_set.dropna()
# print(train_set.isnull().sum())
# print(train_set.shape)
# test_set2 = test_set.dropna()
###### 결측치 처리 4.위치 찾아 제거 #####
 
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


# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],    
hour_bef_precipitation_out_index= outliers(train_set['hour_bef_precipitation'])[0]
hour_bef_windspeed_out_index= outliers(train_set['hour_bef_windspeed'])[0]
hour_bef_humidity_out_index= outliers(train_set['hour_bef_humidity'])[0]
hour_bef_visibility_out_index= outliers(train_set['hour_bef_visibility'])[0]
hour_bef_ozone_out_index= outliers(train_set['hour_bef_ozone'])[0]
hour_bef_pm10_out_index= outliers(train_set['hour_bef_visibility'])[0]
hour_bef_pm25_out_index= outliers(train_set['hour_bef_pm2.5'])[0]
# print(train_set2.loc[hour_bef_precipitation_out_index,'hour_bef_precipitation'])
lead_outlier_index = np.concatenate((hour_bef_precipitation_out_index,
                                     hour_bef_windspeed_out_index,
                                     hour_bef_humidity_out_index,
                                     hour_bef_visibility_out_index,
                                     hour_bef_ozone_out_index,
                                     hour_bef_pm10_out_index,
                                     hour_bef_pm25_out_index),axis=None)
print(len(lead_outlier_index)) #161개 
print(lead_outlier_index)
lead_not_outlier_index = []
for i in train_set.index:
    if i not in lead_outlier_index :
        lead_not_outlier_index.append(i)
train_set_clean = train_set.loc[lead_not_outlier_index]      
train_set_clean = train_set_clean.reset_index(drop=True)
print(train_set_clean)
########################################### test  데이터
# hour_bef_precipitation_out_index2= outliers(test_set['hour_bef_precipitation'])[0]
# hour_bef_windspeed_out_index2= outliers(test_set['hour_bef_windspeed'])[0]
# hour_bef_humidity_out_index2= outliers(test_set['hour_bef_humidity'])[0]
# hour_bef_visibility_out_index2= outliers(test_set['hour_bef_visibility'])[0]
# hour_bef_ozone_out_index2= outliers(test_set['hour_bef_ozone'])[0]
# hour_bef_pm10_out_index2= outliers(test_set['hour_bef_visibility'])[0]
# hour_bef_pm25_out_index2= outliers(test_set['hour_bef_pm2.5'])[0]
# # print(test_set.loc[hour_bef_precipitation_out_index2,'hour_bef_precipitation'])
# lead_outlier_index2 = np.concatenate((hour_bef_precipitation_out_index2,
#                                      hour_bef_windspeed_out_index2,
#                                      hour_bef_humidity_out_index2,
#                                      hour_bef_visibility_out_index2,
#                                      hour_bef_ozone_out_index2,
#                                      hour_bef_pm10_out_index2,
#                                      hour_bef_pm25_out_index2),axis=None)
# print(len(lead_outlier_index2)) #161개 
# print(lead_outlier_index2)
# lead_not_outlier_index2 = []
# for i in test_set.index:
#     if i not in lead_outlier_index2 :
#         lead_not_outlier_index2.append(i)
# test_set_clean = test_set.loc[lead_not_outlier_index2]      
# test_set_clean = test_set_clean.reset_index(drop=True)
# print(test_set_clean)
#################################################



x = train_set_clean.drop(['count'],axis=1) #axis는 컬럼 
print(x.columns)
print(x.shape) #(1459, 9)

y = train_set_clean['count']
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,KFold
from xgboost import XGBClassifier,XGBRegressor
x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.7, random_state=1234)

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
            #   'max_depth': [None,3,4,5,6,7],
              'gamma' : [0,0.1,0.3,0.5,0.7,0.8,0.9,1],
            #   'min_child_weight' : [1],
            #   'subsample' : [1],
            #   'colsample_bytree' : [0.5],
            #   'colsample_bylevel': [1],
            #   'colsample_bynode': [1],
            #   'alpha' : [0],
            #   'lambda' : [0]
              } # 디폴트 6 
# 통상 max_depth의 디폴트인 6보다 작을 파라미터를 줄 때 성능이 좋다 -> 너무 깊어지면 훈련 데이터에 특화되어 과적합이 될 수 있다.
# 통상 min_depth의 디폴트인 6보다 큰 파라미터를 줄 때 성능이 좋다


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
print('걸린 시간 : ',end_time)
#############################################
# 최적의 매개변수 :  {'n_estimators': 100}
# 최상의 점수 :  0.7358984636021871
# model.score : 0.7861693605495705
# 걸린 시간 :  4.022421360015869
#############################################
# 최적의 매개변수 :  {'learning_rate': 0.1, 'n_estimators': 100}
# 최상의 점수 :  0.7371412466144399
# model.score : 0.8116761666300953
# 걸린 시간 :  3.11710786819458
#############################################
# 최적의 매개변수 :  {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}
# 최상의 점수 :  0.752499485599918
# model.score : 0.7817660603161196
# 걸린 시간 :  3.0197927951812744
#############################################
# 최적의 매개변수 :  {'gamma': 0.8, 'learning_rate': 0.1, 'n_estimators': 100}
# 최상의 점수 :  0.7395168530914326
# model.score : 0.8121190578078334
# 걸린 시간 :  3.17083740234375

