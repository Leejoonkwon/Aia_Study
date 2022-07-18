import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.keras import utils
import os
import matplotlib
from sklearn.preprocessing import LabelEncoder
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False

path = './_data/test_amore_0718/' # ".은 현재 폴더"
# Amo1 = pd.read_csv(path + '아모레220718.csv' ,sep='\t',engine='python',encoding='CP949')
Amo = pd.read_csv(path + '아모레220718.csv',thousands=',')

Sam = pd.read_csv(path + '삼성전자220718.csv',thousands=',')

# print(Amo) #[3180 rows x 17 columns]
# print(Sam) #[3040 rows x 17 columns]
# print(Amo.head())
# print(Amo.describe().transpose())
# print(Amo['시가'].corr()) # 상관 계수 확인

# Amo["시가"].plot(figsize=(12,6)) 
# plt.show() # 'T (degC)'열의 전체 데이터 시각화

# plt.figure(figsize=(20,10),dpi=120)
# plt.plot(Amo['시가'][0:6*24*365],color="black",linewidth=0.2)
# plt.show() # 'T (degC)'열의 1년간 데이터 추이 시각화

# print(Amo.info()) #[3180 rows x 17 columns] objetct 14
print(Sam.info()) #(3040 rows x 17 columns] objetct 14


# Amo = Amo.drop([1773,1774,1775,1776,1777,1778,1779,1780,1781,1782,1783], axis=0)

print(Sam.shape) #3037,17

# Sam.at[:1036, '시가'] =1
# print(Sam['시가'])
print(Amo) #2018/05/04
Amo.at[1035:,'시가'] = 0
print(Amo) #2018/05/04


Amo.index = pd.to_datetime(Amo['일자'],
                            format = "%Y/%m/%d") 
Sam.index = pd.to_datetime(Sam['일자'],
                            format = "%Y/%m/%d") 
Sam = Sam[Sam['시가'] < 100000] #[1035 rows x 17 columns]
print(Sam.shape)
print(Sam)
Amo = Amo[Amo['시가'] > 100] #[1035 rows x 17 columns]
print(Amo.shape)
print(Amo) #2018/05/04

# data에 index 열을 Date Time에 연,월,일,시간,분,초를 각각 문자열로 인식해 대체합니다.
# print(Amo.info()) #(420551, 15) DatetimeIndex: 3180 entries, 2022-07-18 to 2009-09-01
# cols = ['시가','고가','저가','종가','전일비','Unnamed: 6','거래량','금액(백만)','개인','기관','외인(수량)','외국계','프로그램']
# for col in cols:
#     le = LabelEncoder()
#     Amo[col]=le.fit_transform(Amo[col])
#     Sam[col]=le.fit_transform(Sam[col])
# print(Amo) 
# print(Amo.info())

Amo = Amo.rename(columns={'Unnamed: 6':'증감량'})
Sam = Sam.rename(columns={'Unnamed: 6':'증감량'})

print(Amo) #[70067 rows x 15 columns] 중복되는 값은 제거한다 행이 70091->에서 70067로 줄어든 것을 확인

Amo1 = Amo.drop([ '전일비', '금액(백만)','신용비','개인','기관','외인(수량)','외국계','프로그램','외인비'],axis=1) #axis는 컬럼 
Sam1 = Sam.drop([ '전일비', '금액(백만)','신용비','개인','기관','외인(수량)','외국계','프로그램','외인비'],axis=1) #axis는 컬럼 
# Sam1 = Sam['시가','고가','저가','종가','증감량','등락률','거래량']
# 삼성전자 2018 05 04부터 액면 분할 시가 저가 고가 종가 *50배 필요

def generator(data, window, offset):
    gen = data.to_numpy() #데이터 프레임을 배열객체로 반환
    x1 = []
    y = []
    for i in range(len(gen)-window-offset): # 420522
        row = [[a] for a in gen[i:i+window]] #행
        x1.append(row)
        label = gen[i+window+offset-1]
        y.append(label)
    return np.array(x1), np.array(y)
WINDOW = 5
OFFSET = 7

x1, y1 = generator(Amo1, WINDOW, OFFSET)
x2, y2 = generator(Sam1, WINDOW, OFFSET)
print(x1,x1.shape) #(3168, 5, 1, 8)
print(x2,x2.shape) #(3028, 5, 1, 8)

# print(y,y.shape) #(70038,)

#시계열 데이터의 특성 상 연속성을 위해서 train_test_split에 셔플을 배제하기 위해
#위 명령어로 정의한다.suffle을 False로 놓고 해도 될지는 모르겠다.
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.layers import InputLayer
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
x1_train,x1_test,x2_train,x2_test,y1_train,y1_test,y2_train,y2_test =train_test_split(x1,x2,y1,y2,shuffle=False,train_size=0.89)
print(x1_train,x1_train.shape) #(3168, 5, 1, 8)
print(x1_test,x1_test.shape) #(3168, 5, 1, 8)
print(x2_train,x2_train.shape) #(3168, 5, 1, 8)
print(x2_test,x2_test.shape) #(3168, 5, 1, 8)


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
'''