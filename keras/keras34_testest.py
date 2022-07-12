#  과제
# activation : sigmoid,relu,linear
# metrics 추가
# EarlyStopping  넣고
# 성능비교
# 감상문 2줄이상!
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten
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

combine = (train_set,test_set)
# data['tempband'] = pd.cut(data['Temperature'], 3)
# print(data['tempband'])
# Categories (3, interval[float64, right]): [(-2.162, 32.007] < (32.007, 66.073] < (66.073, 100.14]]
for dataset in combine:
  dataset.loc[ dataset['Temperature'] <= 32.007, 'Temperature'] = 0
  dataset.loc[(dataset['Temperature'] > 32.007) & (dataset['Temperature'] <=66.073), 'Temperature'] = 1
  dataset.loc[(dataset['Temperature'] > 66.073) & (dataset['Temperature'] <= 100.14), 'Temperature'] = 2
 
# data['Unemplband'] = pd.cut(data['Unemployment'], 3)
# print(data['Unemplband'])
# Categories (3, interval[float64, right]): [(3.869, 7.357] < (7.357, 10.835] < (10.835, 14.313]]
  
for dataset in combine:
  dataset.loc[ dataset['Unemployment'] <= 7.357, 'Unemployment'] = 0
  dataset.loc[(dataset['Unemployment'] > 7.357) & (dataset['Unemployment'] <= 10.835), 'Unemployment'] = 1
  dataset.loc[(dataset['Unemployment'] > 10.835) & (dataset['Unemployment'] <= 14.313), 'Unemployment'] = 2
  
# data['Fuelband'] = pd.cut(data['Fuel_Price'], 3)
# print(data['Fuelband'])
# Categories (3, interval[float64, right]): [(2.47, 3.137] < (3.137, 3.803] < (3.803, 4.468]]


for dataset in combine:
  dataset.loc[ dataset['Fuel_Price'] <= 3.137, 'Fuel_Price'] = 0
  dataset.loc[(dataset['Fuel_Price'] > 3.137) & (dataset['Fuel_Price'] <= 3.803), 'Fuel_Price'] = 1
  dataset.loc[(dataset['Fuel_Price'] > 3.803) & (dataset['Fuel_Price'] <=4.468), 'Fuel_Price'] = 2

# print(train_set[['Fuel_Price', 'Weekly_Sales']].groupby(['Fuel_Price'], as_index=False).mean().sort_values(by='Weekly_Sales', ascending=False))
# # 기름값이 4.069보다 클 때 매출이 가장 낮음 
# print(train_set[['Unemployment', 'Weekly_Sales']].groupby(['Unemployment'], as_index=False).mean().sort_values(by='Weekly_Sales', ascending=False))
# # 실업율이 5.966%이하일 때 매출 가장 높음 
# print(train_set[['IsHoliday', 'Weekly_Sales']].groupby(['IsHoliday'], as_index=False).mean().sort_values(by='Weekly_Sales', ascending=False))
# # 공휴일에 따라 매출 영향 약 7%로 크진 않다.
# print(train_set[['year', 'Weekly_Sales']].groupby(['year'], as_index=False).mean().sort_values(by='Weekly_Sales', ascending=False))
# # year의 평균값은 큰 차이가 없음 

print(test_set.shape) #(180,11) #train_set과 열 값이 '1'차이 나는 건 count를 제외했기 때문이다.예측 단계에서 값을 대입

x = train_set.drop(['Weekly_Sales','year','day'], axis=1) #axis는 컬럼 

print(x) #(6435, 22)

print(x.shape) #

y = train_set['Weekly_Sales']

print(y.shape) # (6255,)
print(test_set) # [180 rows x 14 columns]


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.91, shuffle = True, random_state =100)
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


# print(x_train.shape) # (5566, 13)
# print(x_test.shape) # (689, 13)

# print(test_set.shape) # (180, 14)
# print(test_set) # (180, 14)
test_set = test_set.drop(['Weekly_Sales','year','day'], axis=1)
print(test_set) # (180, 13)

#2. 모델구성
model = Sequential()
model.add(Dense(100, activation='swish',input_dim=11))
model.add(Dropout(0.2))
model.add(Dense(100, activation='swish'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='swish'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='swish'))
model.add(Dense(1, activation='swish'))
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

hist = model.fit(x_train, y_train, epochs=850, batch_size=500, 
                validation_split=0.3,
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
y_predict = model.predict(x_test)
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'],marker='.',c='red',label='loss') #순차적으로 출력이므로  y값 지정 필요 x
plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss')
plt.grid()
plt.title('영어싫어') #맥플러립 한글 깨짐 현상 알아서 해결해라 
plt.ylabel('loss')
plt.xlabel('epochs')
# plt.legend(loc='upper right')
plt.legend()
plt.show()

def RMSE(y_test, y_predict):
     return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE :",rmse)  


y_summit = model.predict(test_set)
submission['Weekly_Sales'] = y_summit
submission.to_csv('test25.csv',index=True)


# scaler = MinMaxScaler()


# scaler = StandardScaler()


# scaler = MaxAbsScaler()


# scaler = RobustScaler()
# loss : 107808.859375
# RMSE : 180006.59736279314

