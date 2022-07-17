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

data['T (degC)'].plot(figsize=(12,6)) # 'T (degC)'열의 전체 데이터 시각화
# plt.show() 

plt.figure(figsize=(20,10),dpi=120)
plt.plot(data['T (degC)'][0:6*24*365],color="black",linewidth=0.2)
# plt.show()

data.index = pd.to_datetime(data['Date Time'],
                            format = "%d.%m.%Y %H:%M:%S")
# data에 index 열을 Date Time에 연,월,일,시간,분,초를 각각 문자열로 인식해 대체합니다.
hourly = data[5::6]
# 5번째 인수의 데이터를 시작으로 6번째에 위치한 데이터만 반환합니다. 
# [5::6]으로 자른 이유는 데이터를 시간단위로 분할하기 위해서다.
#예) 데이터가 [1,2,3,4,5,6,7,8,9,10]을  [1:3]으로 자른다면 [2,5,8]이 된다.
hourly=hourly.drop_duplicates()
#[70067 rows x 15 columns] 중복되는 값은 제거한다 행이 70091->에서 70067로 줄어든 것을 확인
hourly.duplicated().sum()
# 0 중복된 값이 없는 것을 확인
daily = data['T (degC)'].resample('1D').mean().interpolate('linear')
#resample은 본인이 가진 데이터 중 원하는 값만 뽑아냄 시계열 데이터에서 자주 활용 '1D'는 단위 구간을 1일로 설정
daily[0:365].plot()
# plt.show() # 월별 온도 확인
hourly_temp = hourly['T (degC)']
len(hourly_temp) #70067

def generator(data, window, offset):
    gen = data.to_numpy() #데이터 프레임을 배열객체로 반환
    x = []
    y = []
    for i in range(len(gen)-window-offset): #range("")는 ("")값만큼 반복하겠다는 의미
        row = [[a] for a in gen[i:i+window]]
        x.append(row)
        label = gen[i+window+offset-1]
        y.append(label)
    return np.array(x), np.array(y)
WINDOW = 5
OFFSET = 24
x, y = generator(hourly_temp, WINDOW, OFFSET)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=False,train_size=0.91)
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
# scaler.transform(x_train)
x_train = x_train.reshape(63734,5)
x_test = x_test.reshape(6304,5)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# # X_train, y_train = X[:60000], y[:60000]
# # X_val, y_val = X[60000:65000], y[60000:65000]
# # X_test, y_test = X[65000:], y[65000:]
# print(x_train.shape,x_test.shape) #(63734, 5, 1) (6304, 5, 1)
# print(y_train.shape,y_test.shape) #(63734,) (6304,)
x_train = x_train.reshape(63734,5,1)
x_test = x_test.reshape(6304,5,1)
#시계열 데이터의 특성 상 연속성을 위해서 train_test_split에 셔플을 배제하기 위해
#위 명령어로 정의한다.suffle을 False로 놓고 해도 될지는 모르겠다.

from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import LSTM,Conv1D,Flatten,Reshape
from tensorflow.python.keras.layers import InputLayer,Input,Dense
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.callbacks import ModelCheckpoint
#2. 모델구성
input1 = Input(shape=(WINDOW,1),name='input1') #(N,2)
dense1 = Conv1D(64,2,activation='relu',name='jk1')(input1)
dense2 = LSTM(100,activation='relu',name='jk2')(dense1)
dense3 = Dense(64,activation='relu',name='jk3')(dense2)
dense4 = Dense(64,activation='relu',name='jk4')(dense3)
dense5 = Dense(32,activation='relu',name='jk5')(dense4)
dense6 = Dense(32,activation='relu',name='jk6')(dense5)
dense7 = Dense(32,activation='relu',name='jk7')(dense6)
output1 = Dense(1,activation='relu',name='out_jk1')(dense7)
model = Model(inputs=input1, outputs=output1)
model.summary()
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
earlyStopping = EarlyStopping(monitor='loss', patience=10, mode='min', 
                              verbose=1,restore_best_weights=True)
#3. 컴파일,훈련
model.compile(loss='mae', optimizer='Adam')
model.fit(x_train, y_train, validation_split=0.25,
          epochs=50,batch_size=8192,
          callbacks=[earlyStopping]
          ,verbose=2)

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print("loss :",loss)
# loss : 2.3783624172210693

test_predictions = model.predict(x_test).flatten()

result = pd.DataFrame(data={'Predicted': test_predictions, 'Real':y_test})
plt.figure(figsize=(20,7.5),dpi=120)
plt.plot(result['Predicted'][:300], "-g", label="Predicted")
plt.plot(result['Real'][:300], "-r", label="Real")
plt.legend(loc='best')
result['Predicted'] = result['Predicted'].shift(-OFFSET)
result.drop(result.tail(OFFSET).index,inplace = True)
print(result)

#loss : 2.3984692096710205

####LSTM
# loss : 2.3855671882629395