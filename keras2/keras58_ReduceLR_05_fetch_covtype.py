# [과제] 맹그러서 속도 비교
# gpu 와 cpu
import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten,LSTM,Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.metrics import accuracy_score
import pandas as pd 
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if(gpus) : 
    print("쥐피유 돈다")
    aaa = 'gpu'
else:
    print("쥐피유 안도라")
    bbb = 'cpu'

#1.데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(581012, 54) (581012,)

# from sklearn.preprocessing import OneHotEncoder
print('y의 라벨값 :', np.unique(y,return_counts=True))
# y의 라벨값 : (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
    #   dtype=int64))


###########(pandas 버전 원핫인코딩)###############
y_class = pd.get_dummies((y))
print(y_class.shape) # (581012, 7)

# 해당 기능을 통해 y값을 클래스 수에 맞는 열로 늘리는 원핫 인코딩 처리를 한다.
#1개의 컬럼으로 [0,1,2] 였던 값을 ([1,0,0],[0,1,0],[0,0,1]과 같은 shape로 만들어줌)

###########(sklearn 버전 원핫인코딩)###############
#from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(sparse=False)
# # fit_transform은 train에만 사용하고 test에는 학습된 인코더에 fit만 해야한다
# train_cat = ohe.fit_transform(train[['cat1']])
# train_cat


# num = num.shape[0]
# print(num)

# y = np.eye(num)[data]
# print(x)
# print(y)



# print(x.shape, y.shape) #(581012, 54) (581012,)

x_train, x_test, y_train, y_test = train_test_split(x,y_class, test_size=0.15,shuffle=True ,random_state=100)
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#셔플을 False 할 경우 순차적으로 스플릿하다보니 훈련에서는 나오지 않는 값이 생겨 정확도가 떨어진다.
#디폴트 값인  shuffle=True 를 통해 정확도를 올린다.
print(y_train,y_test)

print(x_train.shape) #(493860, 54)
print(x_test.shape) #(87152, 54)

x_train = x_train.reshape(493860, 54,1)
x_test = x_test.reshape(87152, 54,1)



#2. 모델 구성

model = Sequential()
# model.add(LSTM(5,input_shape=(54,1)))
model.add(Conv1D(10,2,input_shape=(54,1)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))
#다중 분류로 나오는 아웃풋 노드의 개수는 y 값의 클래스의 수와 같다.활성화함수 'softmax'를 통해 
# 아웃풋의 합은 1이 된다.
import datetime
date = datetime.datetime.now()
print(date)

date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)

import time
start_time = time.time()

# #3. 컴파일,훈련
filepath = './_ModelCheckPoint/K24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
#04d :                  4f : 
earlyStopping = EarlyStopping(monitor='loss', patience=30, mode='min', 
                              verbose=1,restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
#                       save_best_only=True, 
#                       filepath="".join([filepath,'k25_', date, '_fetcg_covtype_', filename])
#                     )
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, batch_size=250000, 
                validation_split=0.3,
                callbacks = [earlyStopping],
                verbose=2
                )
#다중 분류 모델은 'categorical_crossentropy'만 사용한다 !!!!

#4.  평가,예측

loss,acc = model.evaluate(x_test,y_test)
print('loss :',loss)



y_predict = model.predict(x_test)

print(y_predict) 

# y_test = np.argmax(y_test,axis=1)
import tensorflow as tf
y_test = tf.argmax(y_test,axis=1)
y_predict = tf.argmax(y_predict,axis=1)
#pandas 에서 인코딩 진행시 argmax는 tensorflow 에서 임포트한다.
print(y_predict) #(87152, )
# print(y_test.shape) #(87152,7)
# y_test와 y_predict의  shape가 일치해야한다.



acc = accuracy_score(y_test, y_predict)
print('acc 스코어 :', acc)
end_time =time.time()-start_time
print("걸린 시간 :",end_time)
# drop 아웃 전 
# loss : 0.12787137925624847
# acc 스코어 : 0.9533573526711951
# drop 아웃 후
# loss : 0.17082029581069946
# acc 스코어 : 0.9304777859372132

#cnn dnn 후
# loss : 0.157871812582016
# acc 스코어 : 0.9382458233890215
######LSTM
# loss : 1.3015809059143066
# accuracy: 0.4871
######Conv1d + LR Reduce

