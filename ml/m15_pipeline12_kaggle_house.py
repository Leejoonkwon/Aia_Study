
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
from sklearn.svm import LinearSVC 
from sklearn.svm import LinearSVR 
from sklearn.preprocessing import MinMaxScaler,StandardScaler

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.8
       ,random_state=1234,shuffle=True)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train) 
# x_test = scaler.transform(x_test)

#2. 모델 
from sklearn.svm import LinearSVC, SVC 
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline 

# model = SVC()
# model = make_pipeline(MinMaxScaler(),SVC())
model = make_pipeline(StandardScaler(),RandomForestRegressor())
# 모델 정의와 스케일링을 정의해주지 않아도  fit에서 fit_transform이 적용된다.

#3. 훈련 
model.fit(x_train,y_train)

#4. 평가,예측
result = model.score(x_test,y_test)
# 모델 정의와 상관없이 model로 정의된 기능에 x_test에 transform이 적용된다

print('model.score :',result)
########### ML 시
# results : -3.640761034476303
