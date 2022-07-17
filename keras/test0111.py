import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import os

path = './_data/kaggle_jena/' # ".은 현재 폴더"
data = pd.read_csv(path + 'jena_climate_2009_2016.csv' )
# print(data) #[420551 rows x 15 columns]

# print(data.corr()['T (degC)']) # 상관 계수 확인 
# p (mbar)          -0.045375
# T (degC)           1.000000
# Tpot (K)           0.996827
# Tdew (degC)        0.895708
# rh (%)            -0.572416
# VPmax (mbar)       0.951113
# VPact (mbar)       0.867673
# VPdef (mbar)       0.761744
# sh (g/kg)          0.866755
# H2OC (mmol/mol)    0.867177
# rho (g/m**3)      -0.963410
# wv (m/s)          -0.004689
# max. wv (m/s)     -0.002871
# wd (deg)           0.038732

data['T (degC)'].plot(figsize=(12,6)) 
# plt.show() # 'T (degC)'열의 전체 데이터 시각화

plt.figure(figsize=(20,10),dpi=120)
plt.plot(data['T (degC)'][0:6*24*365],color="black",linewidth=0.2)
# plt.show() # 'T (degC)'열의 1년간 데이터 추이 시각화
print(data) #(420551, 15)
print(data.info()) #(420551, 15)
data.index = pd.to_datetime(data['Date Time'],
                            format = "%d.%m.%Y %H:%M:%S") 
# data에 index 열을 Date Time에 연,월,일,시간,분,초를 각각 문자열로 인식해 대체합니다.
print(data.info()) #(420551, 15) DatetimeIndex: 420551 entries, 2009-01-01 00:10:00 to 2017-01-01 00:00:00

hourly = data[5::6] 
# 5번째 인수의 데이터를 시작으로 6번째에 위치한 데이터만 반환합니다. 
# [5::6]으로 자른 이유는 데이터를 시간단위로 분할하기 위해서다.
#예) 데이터가 [1,2,3,4,5,6,7,8,9,10]을  [1:3]으로 자른다면 [2,5,8]이 된다.
print(hourly) #(70091, 15)
# print(hourly.shape) #(70091, 15)


hourly = hourly.drop_duplicates()
print(hourly) #[70067 rows x 15 columns] 중복되는 값은 제거한다 행이 70091->에서 70067로 줄어든 것을 확인
hourly.duplicated().sum() 

daily = data['T (degC)'].resample('1D').mean().interpolate('linear')
#resample은 본인이 가진 데이터 중 원하는 값만 뽑아냄 시계열 데이터에서 자주 활용 '1D'는 단위 구간을 1일로 설정
daily[0:365].plot()
# plt.show() # 월별 온도 확인

hourly_temp = hourly['T (degC)']
len(hourly_temp) 
print(len(hourly_temp)) #70067

def generator(data, window, offset):
    gen = data.to_numpy() #데이터 프레임을 배열객체로 반환
    X = []
    y = []
    for i in range(len(gen)-window-offset): # 420522
        row = [[a] for a in gen[i:i+window]] #행
        X.append(row)
        label = gen[i+window+offset-1]
        y.append(label)
    return np.array(X), np.array(y)
WINDOW = 5
OFFSET = 24

X, y = generator(hourly_temp, WINDOW, OFFSET)
# print(X,X.shape) #(70038, 5, 1)
# print(y,y.shape) #(70038,)
gen = data.to_numpy()
label = gen[0+WINDOW+OFFSET-1] 
#['01.01.2009 04:50:00' 997.37 -9.47 263.89 -10.46 92.4 2.97 2.75 0.23 1.72
#2.75 1316.25 0.37 0.75 125.8] (15,)
print(label,label.shape)# (15,)


X_train, y_train = X[:60000], y[:60000]
X_val, y_val = X[60000:65000], y[60000:65000]
X_test, y_test = X[65000:], y[65000:]
print(X_train,y_train)
print(X_train.shape,y_train.shape) #(60000, 5, 1) (60000,)


#시계열 데이터의 특성 상 연속성을 위해서 train_test_split에 셔플을 배제하기 위해
#위 명령어로 정의한다.suffle을 False로 놓고 해도 될지는 모르겠다.
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.layers import InputLayer
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


#2. 모델구성
model1 = Sequential()
model1.add(InputLayer((WINDOW, 1)))
model1.add(LSTM(100))
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear'))

model1.summary()

#3. 컴파일,훈련
model1.compile(loss='mae', optimizer='Adam')
model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

#4. 평가,예측
loss = model1.evaluate(X_test, y_test)
print("loss :",loss)
test_predictions = model1.predict(X_test).flatten()

result = pd.DataFrame(data={'Predicted': test_predictions, 'Real':y_test})
plt.figure(figsize=(20,7.5),dpi=120)
plt.plot(result['Predicted'][:300], "-g", label="Predicted")
plt.plot(result['Real'][:300], "-r", label="Real")
plt.legend(loc='best')
result['Predicted'] = result['Predicted'].shift(-OFFSET)
result.drop(result.tail(OFFSET).index,inplace = True)
print(result)
#loss : 2.3984692096710205
