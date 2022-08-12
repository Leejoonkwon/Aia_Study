
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


# print(x.columns)
# print(x.shape) #(10886, 8)

y = train_set_clean['count']

from sklearn.model_selection import KFold,cross_val_score,cross_val_predict

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,KFold,StratifiedKFold

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.8
       ,random_state=1234,shuffle=True)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train) 
# x_test = scaler.transform(x_test)

#2. 모델 
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
    
# 최적의 매개변수 : RandomForestRegressor(max_depth=8, min_samples_leaf=3, n_jobs=2)
# 최적의 파라미터 : {'max_depth': 8, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 100, 'n_jobs': 2}   
# best_score : 0.35497017372207595
# model_score : 0.33946365731794803
# accuracy_score : 0.33946365731794803
# 최적 튠  ACC : 0.33946365731794803
# 걸린 시간 : 62.29 초      
#============= pipe HalvingGridSearchCV
# 최적의 매개변수 : Pipeline(steps=[('minmax', MinMaxScaler()),
#                 ('RF',
#                  RandomForestRegressor(max_depth=8, min_samples_leaf=7,
#                                        min_samples_split=4, n_estimators=400,
#                                        n_jobs=4))])
# 최적의 파라미터 : {'RF__max_depth': 8, 'RF__min_samples_leaf': 7, 'RF__min_samples_split': 4, 
# 'RF__n_estimators': 400, 'RF__n_jobs': 4}      
# best_score : 0.3453183955420628
# model_score : 0.36446649478414794
# accuracy_score : 0.36446649478414794
# 최적 튠  ACC : 0.36446649478414794
# 걸린 시간 : 41.58 초
#============= pipe GridSearchCV
# 최적의 매개변수 : Pipeline(steps=[('minmax', MinMaxScaler()),
#                 ('RF',
#                  RandomForestRegressor(max_depth=8, min_samples_leaf=3,
#                                        n_estimators=200, n_jobs=-1))])
# 최적의 파라미터 : {'RF__max_depth': 8, 'RF__min_samples_leaf': 3, 'RF__min_samples_split': 2, 
# 'RF__n_estimators': 200, 'RF__n_jobs': -1}     
# best_score : 0.3485804362554507
# model_score : 0.36661184664654445
# accuracy_score : 0.36661184664654445
# 최적 튠  ACC : 0.36661184664654445
# 걸린 시간 : 63.97 초
#============= pipe RandomizedSearchCV
# 최적의 매개변수 : Pipeline(steps=[('minmax', MinMaxScaler()),
#                 ('RF',
#                  RandomForestRegressor(max_depth=8, min_samples_leaf=3,
#                                        n_estimators=200, n_jobs=-1))])
# 최적의 파라미터 : {'RF__n_jobs': -1, 'RF__n_estimators': 200, 'RF__min_samples_split': 2, 'RF__min_samples_leaf': 3, 'RF__max_depth': 8}     
# best_score : 0.34827446376168886
# model_score : 0.36599775368477017
# accuracy_score : 0.36599775368477017
# 최적 튠  ACC : 0.36599775368477017
# 걸린 시간 : 11.36 초
#=================  결측치 중위 처리  =============  
# knn-0.24136135449131146
# svr-0.19683139616790846
# tree--0.12026844846386475
# forest-0.321936838554558
# linear-0.2584876622834902
# xgb-0.3502573587090576
#=================  결측치 interpolate 처리  =============  
# knn-0.24136135449131146
# svr-0.19683139616790846
# tree--0.11376865104990008
# forest-0.3279615739262881
# linear-0.2584876622834902
# xgb-0.3502573587090576
#=================  결측치 mean 처리  ============= 
# knn-0.24136135449131146
# svr-0.19683139616790846
# tree--0.13303611825704742
# forest-0.3261881704467139
# linear-0.2584876622834902
# xgb-0.3502573587090576

