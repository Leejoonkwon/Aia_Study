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
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,KFold,StratifiedKFold
x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.7, random_state=1234)

# 2. 모델구성
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor #공부하자 
from sklearn.ensemble import RandomForestRegressor #공부하자 
from sklearn.linear_model import LinearRegression 
from sklearn.svm import SVR
# models = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]
def models(model):
    if model == 'knn':
        mod = KNeighborsRegressor()
    elif model == 'svr':
        mod = SVR()
    elif model == 'tree':
        mod =  DecisionTreeRegressor()
    elif model == 'forest':
        mod =  RandomForestRegressor()
    elif model == 'linear':
        mod =  LinearRegression ()    
    elif model == 'xgb':
        mod =  XGBRegressor ()        
    return mod
model_list = ['knn', 'svr',  'tree', 'forest','linear','xgb']
empty_list = [] #empty list for progress bar in tqdm library
for model in (model_list):
    empty_list.append(model) # fill empty_list to fill progress bar
    #classifier
    clf = models(model)
    #Training
    clf.fit(x_train, y_train) 
    #Predict
    result = clf.score(x_test,y_test)
    pred = clf.predict(x_test) 
    print('{}-{}'.format(model,result))
    
        
#============= pipe HalvingGridSearchCV
# 최적의 매개변수 : Pipeline(steps=[('minmax', MinMaxScaler()),
#                 ('RF',
#                  RandomForestRegressor(max_depth=6, min_samples_leaf=5,
#                                        min_samples_split=3, n_jobs=2))])
# 최적의 파라미터 : {'RF__max_depth': 6, 'RF__min_samples_leaf': 5, 'RF__min_samples_split': 3, 
# 'RF__n_estimators': 100, 'RF__n_jobs': 2}      
# best_score : 0.7507959223647331
# model_score : 0.7591141162607191
# accuracy_score : 0.7591141162607191
# 최적 튠  ACC : 0.7591141162607191
# 걸린 시간 : 25.3 초
#============= pipe GridSearchCV
# 최적의 매개변수 : Pipeline(steps=[('minmax', MinMaxScaler()),
#                 ('RF',
#                  RandomForestRegressor(max_depth=8, min_samples_leaf=3,
#                                        min_samples_split=3, n_estimators=200,
#                                        n_jobs=-1))])
# 최적의 파라미터 : {'RF__max_depth': 8, 'RF__min_samples_leaf': 3, 'RF__min_samples_split': 3, 
# 'RF__n_estimators': 200, 'RF__n_jobs': -1}     
# best_score : 0.7655431824311288
# model_score : 0.7796496700810223
# accuracy_score : 0.7796496700810222
# 최적 튠  ACC : 0.7796496700810222
# 걸린 시간 : 30.97 초

#============= pipe RandomizedSearchCV
# 최적의 매개변수 : Pipeline(steps=[('minmax', MinMaxScaler()),
#                 ('RF',
#                  RandomForestRegressor(max_depth=8, min_samples_leaf=3,
#                                        min_samples_split=3, n_estimators=200,
#                                        n_jobs=-1))])
# 최적의 파라미터 : {'RF__n_jobs': -1, 'RF__n_estimators': 200, 'RF__min_samples_split': 3, 'RF__min_samples_leaf': 3, 'RF__max_depth': 8}     
# best_score : 0.7655627301006306
# model_score : 0.7795832657214612
# accuracy_score : 0.7795832657214612
# 최적 튠  ACC : 0.7795832657214612
# 걸린 시간 : 6.5 초
#=================  결측치 median 처리  =============  
# knn-0.05005421109390007
# svr-0.0371351609788686
# tree-0.5338291078101032
# forest-0.7827754490472193
# linear-0.6110695353409068
# xgb-0.7864238734420762   
#=================  결측치 interpolate 처리  =============  
# knn--0.021160213099664205
# svr-0.0392869877072497
# tree-0.65829171863489
# forest-0.7879504683704567
# linear-0.6131357757567988
# xgb-0.7914603072315469
#=================  결측치 mean 처리  =============  
# knn--0.0029500086857461305
# svr-0.024037253858829266
# tree-0.6058261668039286
# forest-0.7857045058880551
# linear-0.6151305845513807
# xgb-0.7967861479060181
#=================  결측치 drop 처리  =============  


