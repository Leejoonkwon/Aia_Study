#1. 데이터
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense,Input
from tensorflow.python.keras.layers import LSTM,Reshape,Conv2D,Conv1D
from sklearn.model_selection import train_test_split

x1_datasets = np.array([range(100),range(301,401)]) # 삼성전자 종가 ,하이닉스 종가
x2_datasets = np.array([range(101,201),range(411,511),range(150,250)]) # 원유,돈육,밀
x1 = np.transpose(x1_datasets)
x2 = np.transpose(x2_datasets)
print(x1_datasets.shape,x2_datasets.shape) #(100, 2) (100, 3)

y = np.array(range(2001,2101)) 
print(y,y.shape) #금리 (100,)

#2. 모델구성

x1_train,x1_test,x2_train,x2_test,y_train,y_test=train_test_split(x1,x2,y,
                                                                  train_size=0.7,random_state=100)

print(x1_train.shape,x2_train.shape)    #(70, 2) (70, 3)
print(x1_test.shape,x2_test.shape)      #(30, 2) (30, 3)
print(y_train.shape,y_train.shape)      #(70,) (70,)

#2-1. 모델구성1
input1 = Input(shape=(2,)) #(N,2)
dense1 = Dense(64,activation='relu',name='jk1')(input1) # (N,64)
dense2 = Reshape(target_shape=(32,2),name='jk101')(dense1) # (N,64)
dense3 = Conv1D(10,2,name='jk102',activation='relu')(dense2)
dense4 = LSTM(10,name='jk103')(dense3)
dense5 = Dense(64,activation='relu',name='jk2')(dense4) # (N,64)
dense6 = Dense(64,activation='relu',name='jk3')(dense5) # (N,64)
output1 = Dense(10,activation='relu',name='out_ys1')(dense6)

#2-2. 모델구성2
input2 = Input(shape=(3,))
dense11 = Dense(64,activation='relu',name='jk11')(input2) 
dense12 = Reshape(target_shape=(32,2),name='jk104')(dense11)
dense13 = Conv1D(10,2,name='jk105')(dense12)
dense14 = LSTM(10,name='jk106')(dense13)
dense15 = Dense(64,activation='relu',name='jk12')(dense14)
dense16 = Dense(64,activation='relu',name='jk13')(dense15)
dense17 = Dense(32,activation='relu',name='jk14')(dense16)
output2 = Dense(10,activation='relu',name='out_ys2')(dense17)

from tensorflow.python.keras.layers import concatenate,Concatenate
merge1 = concatenate([output1,output2],name= 'mg1')
merge2 = Dense(32,activation='relu',name='mg2')(merge1)
merge3 = Dense(16,activation='relu',name='mg3')(merge2)
merge4 = Dense(16,activation='relu',name='mg4')(merge3)

last_output = Dense(1,name='last')(merge4)
model = Model(inputs=[input1,input2], outputs=last_output)
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')

model.fit([x1_train,x2_train], y_train, epochs=350, 
                batch_size=512,
                validation_split=0.33,
                verbose=2
                )


#4. 평가, 예측
loss = model.evaluate([x1_test,x2_test], y_test)
print('loss :', loss)
# y_predict = model.predict([x1_test,x2_test])
# print(y_predict)
