import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import os
import matplotlib
from sklearn.preprocessing import LabelEncoder
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False

path = './_data/test_amore_0718/' # ".은 현재 폴더"
# Amo1 = pd.read_csv(path + '아모레220718.csv' ,sep='\t',engine='python',encoding='CP949')
Amo = pd.read_csv(path + '아모레220718.csv',thousands=',',encoding='cp949')

Sam = pd.read_csv(path + '삼성전자220718.csv',thousands=',',encoding='cp949')

print(Sam.info()) #(3040 rows x 17 columns] objetct 14



print(Sam.shape) #3037,17

# Sam.at[:1036, '시가'] =1
# print(Sam['시가'])
print(Amo) #2018/05/04
Amo.at[1035:,'시가'] = 0
print(Amo) #2018/05/04


# Amo.index = pd.to_datetime(Amo['일자'],
#                             format = "%Y/%m/%d") 
# Sam.index = pd.to_datetime(Sam['일자'],
#                             format = "%Y/%m/%d") 
Amo['Date'] = pd.to_datetime(Amo['일자'])

Amo['year'] = Amo['Date'].dt.strftime('%Y')
Amo['month'] = Amo['Date'].dt.strftime('%m')
Amo['day'] = Amo['Date'].dt.strftime('%d')
print(Amo)
print(Amo.shape)
Sam['Date'] = pd.to_datetime(Sam['일자'])

Sam['year'] = Sam['Date'].dt.strftime('%Y')
Sam['month'] = Sam['Date'].dt.strftime('%m')
Sam['day'] = Sam['Date'].dt.strftime('%d')



Sam = Sam[Sam['시가'] < 100000] #[1035 rows x 17 columns]
print(Sam.shape)
print(Sam)
Amo = Amo[Amo['시가'] > 100] #[1035 rows x 17 columns]
print(Amo.shape)
print(Amo) #2018/05/04


# data에 index 열을 Date Time에 연,월,일,시간,분,초를 각각 문자열로 인식해 대체합니다.
# print(Amo.info()) #(420551, 15) DatetimeIndex: 3180 entries, 2022-07-18 to 2009-09-01
cols = ['year','month','day']
for col in cols:
    le = LabelEncoder()
    Amo[col]=le.fit_transform(Amo[col])
    Sam[col]=le.fit_transform(Sam[col])
    
Amo = Amo.sort_values(by=['일자'],axis=0,ascending=True)
Sam = Sam.sort_values(by=['일자'],axis=0,ascending=True)

Amo1 = Amo[[ '시가', '고가', '저가', '종가','year','month','day']]
Sam1 = Sam[[ '시가', '고가', '저가', '종가','year','month','day']]
Amo2 = Amo1['종가']

def generator(data, window, offset):
    gen = data.to_numpy() #데이터 프레임을 배열객체로 반환
    X = []
    
    for i in range(len(gen)-window-offset): # 420522
        row = [[a] for a in gen[i:i+window]] #행
        X.append(row)
        
        
    return np.array(X)
WINDOW = 10
OFFSET = 30
print(Amo1.shape) #(1031, 3, 10)

aaa = generator(Amo1,WINDOW,OFFSET) #x1를 위한 데이터 drop 제외 모든 amo에 데이터
bbb = generator(Amo2,WINDOW,OFFSET) #Y를 위한 시가만있는 데이터
x1 = aaa[:,:-3]
y = bbb[:,-3:]
ccc = generator(Sam1,WINDOW,OFFSET) #x2를 위한 데이터 drop 제외 모든 Sam에 데이터
x2 = ccc[:,:-3]

#시계열 데이터의 특성 상 연속성을 위해서 train_test_split에 셔플을 배제하기 위해
#위 명령어로 정의한다.suffle을 False로 놓고 해도 될지는 모르겠다.
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import LSTM,Dense,Dropout,Reshape,Conv1D
from tensorflow.python.keras.layers import Input
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split
x1_train,x1_test,x2_train,x2_test,y_train,y_test =train_test_split(x1,x2,y,shuffle=False,train_size=0.75)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer

x1_train = x1_train.reshape(746,7,7)
x1_test = x1_test.reshape(249,7,7)
x2_train = x2_train.reshape(746,7,7)
x2_test = x2_test.reshape(249,7,7)
print(x1_train,x1_train.shape) #(746, 7, 1,7)
print(x1_test,x1_test.shape) #(249, 7, 1,7)
print(x2_train,x2_train.shape) # (746, 7, 1,7)
print(x2_test,x2_test.shape) # (249, 7, 1,7)
print(y_train.shape) #(746,3,1)


#2-1. 모델구성1
input1 = Input(shape=(7,7)) #(N,2)
dense1 = LSTM(100,activation='relu',name='jk1')(input1)
dense2 = Dropout(0.15)(dense1)
dense3 = Dense(64,activation='relu',name='jk2')(dense2) # (N,64)
dense4 = Dropout(0.15)(dense3)
output1 = Dense(10,activation='linear',name='out_jk1')(dense4)

#2-2. 모델구성2
input2 = Input(shape=(7,7)) #(N,2)
dense4 = LSTM(100,activation='relu',name='jk101')(input2)
dense5 = Dropout(0.15)(dense4)
dense6 = Dense(64,activation='relu',name='jk103')(dense5) 
dense7 = Dropout(0.15)(dense6)
output2 = Dense(10,activation='linear',name='out_jk2')(dense7)

from tensorflow.python.keras.layers import concatenate,Concatenate
merge1 = concatenate([output1,output2],name= 'mg1')
merge2 = Dense(32,activation='relu',name='mg2')(merge1)
merge3 = Dropout(0.15)(merge2)
merge4 = Dense(16,activation='relu',name='mg3')(merge3)
merge5 = Dropout(0.15)(merge4)
merge6 = Dense(16,activation='linear',name='mg4')(merge5)
last_output = Dense(1,name='last')(merge6)
model = Model(inputs=[input1,input2], outputs=last_output)
import datetime
date = datetime.datetime.now()
print(date)

date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)

#3. 컴파일,훈련
filepath = './_ModelCheckPoint/K24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
#04d :                  4f : 
earlyStopping = EarlyStopping(monitor='loss', patience=10, mode='min', 
                              verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
                      save_best_only=True, 
                      filepath="".join([filepath,'k24_', date, '_', filename])
                    )
model.compile(loss='mae', optimizer='Adam')

model.fit([x1_train,x2_train], y_train, 
          validation_split=0.25, 
          epochs=150,verbose=2
          ,batch_size=128
          ,callbacks=[earlyStopping,mcp])
model.save_weights("./_save/keras46_3_save_weights().h5")

#4. 평가,예측
loss = model.evaluate([x1_test,x2_test], y_test)
print("loss :",loss)
print("====================")

y_predict = model.predict([x1_test,x2_test])
print("0720자 종가 :",y_predict[-1])
#스케일링 전
# loss : 1615.2457275390625
# ====================
# 0719자 시가 : [[156286.]]

# loss : 2053.936767578125
# ====================
# 0719자 시가 : [[157145.89]]

# scaler = StandardScaler() 했을 때 
# loss : 432307.78125
# ====================
# 0719자 시가 : [[473035.94]]




