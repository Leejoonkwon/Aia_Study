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
<<<<<<< HEAD:keras/keras46_amore3_ljk_save.py

Sam = pd.read_csv(path + '삼성전자220718.csv',thousands=',',encoding='cp949')

Amo.loc[1035:,'시가'] = 0
=======

Sam = pd.read_csv(path + '삼성전자220718.csv',thousands=',',encoding='cp949')
# print(Amo.corr())

Amo.at[1035:,'시가'] = 0
>>>>>>> 2bec2bb76d3d8977afcd9264af5186df6df717c5:keras/keras45_amore3_ljk_save.py

Amo['Date'] = pd.to_datetime(Amo['일자'])
Amo['year'] = Amo['Date'].dt.strftime('%Y')
Amo['month'] = Amo['Date'].dt.strftime('%m')
Amo['day'] = Amo['Date'].dt.strftime('%d')

Sam['Date'] = pd.to_datetime(Sam['일자'])
Sam['year'] = Sam['Date'].dt.strftime('%Y')
Sam['month'] = Sam['Date'].dt.strftime('%m')
Sam['day'] = Sam['Date'].dt.strftime('%d')



Sam = Sam[Sam['시가'] < 100000] #[1035 rows x 17 columns]
Amo = Amo[Amo['시가'] > 100] #[1035 rows x 17 columns]

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
WINDOW = 6
OFFSET = 20

aaa = generator(Amo1,WINDOW,OFFSET) #x1를 위한 데이터 drop 제외 모든 amo에 데이터
bbb = generator(Amo2,WINDOW,OFFSET) #Y를 위한 시가만있는 데이터
x1 = aaa[:,:-3]
y = bbb[:,-3:]
ccc = generator(Sam1,WINDOW,OFFSET) #x2를 위한 데이터 drop 제외 모든 Sam에 데이터
x2 = ccc[:,:-3]

from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import LSTM,Dense,Dropout,Reshape,Conv1D
from tensorflow.python.keras.layers import Input
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split
x1_train,x1_test,x2_train,x2_test,y_train,y_test =train_test_split(x1,x2,y,shuffle=False,train_size=0.75)

x1_train = x1_train.reshape(756,3,7)
x1_test = x1_test.reshape(253,3,7)
x2_train = x2_train.reshape(756,3,7)
x2_test = x2_test.reshape(253,3,7)
print(x1_train,x1_train.shape) #(756, 3, 1,7)
print(x1_test,x1_test.shape) #(253, 3, 1,7)
print(x2_train,x2_train.shape) # (756, 3, 1,7)
print(x2_test,x2_test.shape) # (253, 3, 1,7)
print(y_train.shape) #(756,3,1)

#2-1. 모델구성1
input1 = Input(shape=(3,7)) #(N,2)
dense1 = LSTM(100,activation='relu',name='jk1')(input1)
dense2 = Dropout(0.15)(dense1)
dense3 = Dense(64,activation='relu',name='jk2')(dense2) # (N,64)
dense4 = Dropout(0.15)(dense3)
output1 = Dense(10,activation='linear',name='out_jk1')(dense4)

#2-2. 모델구성2
input2 = Input(shape=(3,7)) #(N,2)
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
filepath = './_test/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
<<<<<<< HEAD:keras/keras46_amore3_ljk_save.py
#04d :                  4f : 
=======
>>>>>>> 2bec2bb76d3d8977afcd9264af5186df6df717c5:keras/keras45_amore3_ljk_save.py
earlyStopping = EarlyStopping(monitor='loss', patience=30, mode='min', 
                              verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
                      save_best_only=True, 
                      filepath="".join([filepath,'k24_', date, '_', filename])
                    )
model.compile(loss='mae', optimizer='Adam')

model.fit([x1_train,x2_train], y_train, 
          validation_split=0.25, 
          epochs=100,verbose=2
          ,batch_size=256
          ,callbacks=[earlyStopping,mcp])
model.save_weights("./_save/keras46_3_save_weights().h5")

#4. 평가,예측
loss = model.evaluate([x1_test,x2_test], y_test)
print("loss :",loss)
print("====================")

y_predict = model.predict([x1_test,x2_test])
print("0720자 종가 :",y_predict[-1])
# loss : 36247.66796875
# ====================
# 0720자 종가 : [122495.8]

<<<<<<< HEAD:keras/keras46_amore3_ljk_save.py



=======
>>>>>>> 2bec2bb76d3d8977afcd9264af5186df6df717c5:keras/keras45_amore3_ljk_save.py

