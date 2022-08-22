
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


# print(test_set)
# print(train_set.shape) #(1460,76)
# print(test_set.shape) #(1459, 75) #train_set과 열 값이 '1'차이 나는 건 count를 제외했기 때문이다.예측 단계에서 값을 대입

# print(train_set.columns)
# print(train_set.info()) #null은 누락된 값이라고 하고 "결측치"라고도 한다.
# print(train_set.describe()) 


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
Index=['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
       'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
       'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
       'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition']     

    
MSSubClass_out_index= outliers(train_set['MSSubClass'])[0]
MSZoning_out_index= outliers(train_set['MSZoning'])[0]
LotFrontage_out_index= outliers(train_set['LotFrontage'])[0]
Street_out_index= outliers(train_set['Street'])[0]
LotShape_out_index= outliers(train_set['LotShape'])[0]
LandContour_out_index= outliers(train_set['LandContour'])[0]
Utilities_out_index= outliers(train_set['Utilities'])[0]
LotConfig_out_index= outliers(train_set['LotConfig'])[0]
LandSlope_out_index= outliers(train_set['LandSlope'])[0]
Neighborhood_out_index= outliers(train_set['Neighborhood'])[0]

Condition1_out_index= outliers(train_set['Condition1'])[0]
Condition2_out_index= outliers(train_set['Condition2'])[0]
BldgType_out_index= outliers(train_set['BldgType'])[0]
HouseStyle_out_index= outliers(train_set['HouseStyle'])[0]
OverallQual_out_index= outliers(train_set['OverallQual'])[0]
YearBuilt_out_index= outliers(train_set['YearBuilt'])[0]
YearRemodAdd_out_index= outliers(train_set['YearRemodAdd'])[0]
RoofStyle_out_index= outliers(train_set['RoofStyle'])[0]
RoofMatl_out_index= outliers(train_set['RoofMatl'])[0]
Exterior1st_out_index= outliers(train_set['Exterior1st'])[0]
Exterior2nd_out_index= outliers(train_set['Exterior2nd'])[0]
MasVnrType_out_index= outliers(train_set['MasVnrType'])[0]
MasVnrArea_out_index= outliers(train_set['MasVnrArea'])[0]


Exterior2nd_out_index= outliers(train_set['Exterior2nd'])[0]


lead_outlier_index = np.concatenate((MSSubClass_out_index,
                                     MSZoning_out_index,
                                     LotFrontage_out_index,
                                     Street_out_index,
                                     LotShape_out_index,
                                     LandContour_out_index,
                                     Utilities_out_index,
                                     LotConfig_out_index,
                                     LandSlope_out_index,
                                     Neighborhood_out_index),axis=None)
print(len(lead_outlier_index)) #161개 
print(lead_outlier_index)
lead_not_outlier_index = []
for i in train_set.index:
    if i not in lead_outlier_index :
        lead_not_outlier_index.append(i)
train_set_clean = train_set.loc[lead_not_outlier_index]      
train_set_clean = train_set_clean.reset_index(drop=True)
print(train_set_clean)

###### 결측치 처리 1.중위 ##### 
# print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
# train_set = train_set.fillna(train_set.median())
# print(train_set.isnull().sum())
# print(train_set.shape)
# test_set = test_set.fillna(test_set.median())

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
print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
train_set = train_set.dropna()
print(train_set.isnull().sum())
print(train_set.shape)
test_set = test_set.dropna()

x = train_set.drop(['SalePrice'],axis=1) #axis는 컬럼 
print(x.columns)
print(x.shape) #(1460, 75)

y = train_set['SalePrice']
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
              'learning_rate':[0.1,0.2,0.3,0.4],
              'max_depth': [None,3,4,5],
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
print('걸린 시간 : ',end_time)

#============= pipe HalvingGridSearchCV
# 최적의 매개변수 : Pipeline(steps=[('minmax', MinMaxScaler()),
#                 ('RF',
#                  RandomForestRegressor(max_depth=8, min_samples_leaf=3,
#                                        n_estimators=200, n_jobs=2))])
# 최적의 파라미터 : {'RF__max_depth': 8, 'RF__min_samples_leaf': 3, 'RF__min_samples_split': 2, 
# 'RF__n_estimators': 200, 'RF__n_jobs': 2}      
# best_score : 0.8581178318511601
# model_score : 0.863593220963692
# accuracy_score : 0.863593220963692
# 최적 튠  ACC : 0.863593220963692
# 걸린 시간 : 64.59 초
#============= pipe GridSearchCV
# 최적의 매개변수 : Pipeline(steps=[('minmax', MinMaxScaler()),
#                 ('RF',
#                  RandomForestRegressor(max_depth=8, min_samples_leaf=3,
#                                        min_samples_split=3, n_jobs=2))])
# 최적의 파라미터 : {'RF__max_depth': 8, 'RF__min_samples_leaf': 3, 'RF__min_samples_split': 3, 
# 'RF__n_estimators': 100, 'RF__n_jobs': 2}      
# best_score : 0.8604112043120843
# model_score : 0.8664759216302295
# accuracy_score : 0.8664759216302295
# 최적 튠  ACC : 0.8664759216302295
# 걸린 시간 : 65.7 초

# 최적의 매개변수 :  {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 100}
# 최상의 점수 :  0.8623334222203332
# model.score : 0.8400217157746317
# 걸린 시간 :  3.7893898487091064
