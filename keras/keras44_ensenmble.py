#1. 데이터
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense,Input
from tensorflow.python.keras.layers import LSTM,Reshape,Conv2D,Conv1D
from sklearn.model_selection import train_test_split

x1_datasets = np.array([range(100),range(301,401)]) # 삼성전자 종가 ,하이닉스 종가


x1 = np.transpose(x1_datasets)


print(x1_datasets.shape) #(100, 2) (100, 3)

y1 = np.array(range(2001,2101)) 
y2 = np.array(range(201,301)) 

print(y1,y1.shape) #금리 (100,)
print(y2,y2.shape) #환율 (100,)
############ 맹그러라

#2. 모델구성
 # \ 이표시는 ()로 묶지 않아도 다음 줄의 문장이 윗 줄과 이어지게 해주는 약속어
x1_train,x1_test,\
y1_train,y1_test,y2_train,y2_test=train_test_split(x1,y1,y2,
                                                                  train_size=0.7,random_state=100)

# print(x1_train.shape,x2_train.shape)    #(70, 2) (70, 3)
# print(x1_test.shape,x2_test.shape)      #(30, 2) (30, 3)
# print(y1_train.shape,y1_test.shape)      #(70,) (30,)
# print(y2_train.shape,y2_test.shape)      #(70,) (30,)

#2-1. 모델구성1
input1 = Input(shape=(2,)) #(N,2)
dense1 = Dense(64,activation='relu',name='jk1')(input1) # (N,64)
dense2 = Reshape(target_shape=(32,2),name='jk101')(dense1) # (N,64)
dense3 = Conv1D(10,2,name='jk102',activation='relu')(dense2)
# dense4 = LSTM(10,name='jk103')(dense3)
dense4 = Reshape(target_shape=(310,),name='jk103')(dense3) # (N,64)
dense5 = Dense(64,activation='relu',name='jk2')(dense4) # (N,64)
dense6 = Dense(64,activation='relu',name='jk3')(dense5) # (N,64)
output1 = Dense(10,activation='relu',name='out_ys1')(dense6)

# Concatenate 모델 분기!
from tensorflow.python.keras.layers import concatenate,Concatenate
# merge1 = concatenate([output1,output2,output3],name= 'mg1')
merge1 = output1


#2-4. output모델1
merge2 = Dense(32,activation='relu',name='mg2')(merge1)
merge3 = Dense(16,activation='relu',name='mg3')(merge2)
merge4 = Dense(16,activation='relu',name='mg4')(merge3)
last_output1 = Dense(1,name='last1')(merge4)
#2-5. output모델2
merge5 = Dense(32,activation='relu',name='mg5')(merge1)
merge6 = Dense(16,activation='relu',name='mg6')(merge5)
merge7 = Dense(16,activation='relu',name='mg7')(merge6)
last_output2 = Dense(1,name='last2')(merge7)




# 두개 이상의 데이터는 리스트 받아야한다 []
model = Model(inputs=input1, outputs=[last_output1,last_output2])
model.summary()

import time
start_time = time.time()
#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')

hist = model.fit(x1_train, [y1_train,y2_train], 
                epochs=3, 
                batch_size=56,
                validation_split=0.33,
                verbose=2
                )

end_time= time.time()-start_time
#4. 평가, 예측
loss = model.evaluate(x1_test, [y1_test,y2_test])
# loss2 = model.evaluate([x1_test,x2_test,x3_test], y2_test)

print('loss :', loss)
# print('loss :', loss2)

print('걸린 시간 :', end_time)
y_predict = model.predict(x1_test)
print(y_predict)


from sklearn.metrics import r2_score

r2 = r2_score(y1_test,y_predict[0])
r21 = r2_score(y2_test,y_predict[1])


print('r2스코어 :', r2)
print('r2스코어 :', r2)


# loss : [2195.65478515625, 1959.9598388671875, 235.69483947753906]
# 1번째는 last1과 last2의 합계
# 2번째는 last1의 loss
# 3번째는 last2의 loss
