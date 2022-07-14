#  과제
# activation : sigmoid,relu,linear
# metrics 추가
# EarlyStopping  넣고
# 성능비교
# 감상문 2줄이상!
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense,Dropout,LSTM
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
import pandas as pd
import datetime
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold,StratifiedKFold
#1. 데이터
path = './_data/kaggle_shopping/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
print(train_set)

print(train_set.shape) #(6255, 12)

test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)
submission = pd.read_csv(path + 'sample_submission.csv',#예측에서 쓸거야!!
                       index_col=0)
# train_set['Store_prom1']    = train_set.groupby(['Store'])['Promotion1'].transform('mean')
# train_set['Store_prom2']    = train_set.groupby(['Store'])['Promotion2'].transform('mean')
# train_set['Store_prom3']    = train_set.groupby(['Store'])['Promotion3'].transform('mean')
# train_set['Store_prom4']    = train_set.groupby(['Store'])['Promotion4'].transform('mean')
# train_set['Store_prom5']    = train_set.groupby(['Store'])['Promotion5'].transform('mean')

data = pd.concat([train_set,test_set])
data['Date'] = pd.to_datetime(data['Date'])

data['year'] = data['Date'].dt.strftime('%Y')
data['month'] = data['Date'].dt.strftime('%m')
data['day'] = data['Date'].dt.strftime('%d')
print(data)
print(data.shape)
print('=================')

data = data.drop(['Date'], axis=1)
data = data.fillna(0)
print(data)  #[6435 rows x 57 columns]

cols = ['IsHoliday','year','month','day']
for col in cols:
    le = LabelEncoder()
    data[col]=le.fit_transform(data[col])

train_set = data[0:len(train_set)] # train과 test 분리
test_set = data[len(train_set):]

print(train_set.isnull().sum())
print(train_set.shape) #(6255, 14)
print(test_set.shape) #(180, 14)


print(train_set.columns)
# 'Store', 'Date', 'Temperature', 'Fuel_Price', 'Promotion1',
#        'Promotion2', 'Promotion3', 'Promotion4', 'Promotion5', 'Unemployment',
#        'IsHoliday', 'Weekly_Sales'
# print(train_set.info()) #null은 누락된 값이라고 하고 "결측치"라고도 한다.
# print(train_set.describe()) 
print(test_set.columns)
# 'Store', 'Date', 'Temperature', 'Fuel_Price', 'Promotion1',
#        'Promotion2', 'Promotion3', 'Promotion4', 'Promotion5', 'Unemployment',
#        'IsHoliday'

###### 결측치 처리 1.제거##### dropna 사용
print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
# Store              0
# Date               0
# Temperature        0
# Fuel_Price         0
# Promotion1      4153 
# Promotion2      4663
# Promotion3      4370
# Promotion4      4436
# Promotion5      4140
# Unemployment       0
# IsHoliday          0
# Weekly_Sales       0
print(test_set.isnull().sum()) #각 컬럼당 결측치의 합계
# Store             0
# Date              0
# Temperature       0
# Fuel_Price        0
# Promotion1        2
# Promotion2      135
# Promotion3       19
# Promotion4       34
# Promotion5        0
# Unemployment      0
# IsHoliday         0

print(train_set) #[6435 rows x 24 columns]
# train_set = pd.get_dummies(train_set, columns = ['month'])
# test_set = pd.get_dummies(test_set, columns = ['month'])
# test_set = test_set.drop(['Date'], axis=1)

x = train_set.drop(['Weekly_Sales'], axis=1) #axis는 컬럼 

print(x) #(6435, 22)

print(x.shape) #

y = train_set['Weekly_Sales']

print(y.shape) # (6255,)
print(test_set) # [180 rows x 14 columns]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.89, shuffle = True, random_state =100)
print(x_test.shape)

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


print(x_train.shape) # (5566, 13)
print(x_test.shape) # (689, 13)
x_train = x_train.reshape(5566,13,1)
x_test = x_test.reshape(689,13,1)

test_set = test_set.drop(['Weekly_Sales'], axis=1)
print(test_set) # (180,13)

#2. 모델구성
model = Sequential()
model.add(LSTM(units=100, input_length=13,input_dim=1)) #위와 같은 개념
model.add(Dense(100, activation='swish'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='swish'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='swish'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='swish'))
model.add(Dense(1))
import datetime
date = datetime.datetime.now()
print(date)

date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)


# #3. 컴파일,훈련
filepath = './_ModelCheckPoint/K24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
#04d :                  4f : 
earlyStopping = EarlyStopping(monitor='loss', patience=100, mode='min', 
                              verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
                      save_best_only=True, 
                      filepath="".join([filepath,'k25_', date, '_kaggle_shopping', filename])
                    )
model.compile(loss='mae', optimizer='adam')

hist = model.fit(x_train, y_train, epochs=150, batch_size=512, 
                validation_split=0.25,
                callbacks = [earlyStopping],
                verbose=2
                )
print(test_set.shape)

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt

import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
import time
# y_predict = model.predict(x_test)
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'],marker='.',c='red',label='loss') #순차적으로 출력이므로  y값 지정 필요 x
# plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss')
# plt.grid()
# plt.title('영어싫어') #맥플러립 한글 깨짐 현상 알아서 해결해라 
# plt.ylabel('loss')
# plt.xlabel('epochs')
# # plt.legend(loc='upper right')
# plt.legend()
# plt.show()

def RMSE(y_test, y_predict):
     return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE :",rmse)  
# print(test_set.info())
# print(test_set.shape)

# print(y_test)
# print(test_set) #( 180, 13)
# print(test_set.info()) #
# test_set = test_set.astype({'month':'float64'})
# print(test_set.shape) #(180, 11)
# x_test = x_test.reshape(689, 11)
# print(x_train.shape) #(5566, 11, 1, 1)


test_set = test_set.reshape(180,13,1)
y_summit = model.predict(test_set)
submission['Weekly_Sales'] = y_summit
submission.to_csv('test21.csv',index=True)

# MinMaxScaler()
# loss : 126330.3828125
# RMSE : 193888.8899267907

# StandardScaler()
# loss : 99122.25
# RMSE : 180036.57792436687

# scaler = StandardScaler()
# loss : 107203.8984375
# RMSE : 173036.29777622773

# loss : 163955.578125
# RMSE : 261464.49096885187


