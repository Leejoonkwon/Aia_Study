import numpy as np
from sklearn.datasets import load_wine,load_digits
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten,LSTM,Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.metrics import accuracy_score

#1.데이터
datasets = load_digits()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(1797, 64) (1797,)

print(np.unique(y)) # [0 1 2 3 4 5 6 7 8 9]
print(datasets.DESCR)
print(datasets.feature_names)
x = datasets.data
y = datasets['target']
# from sklearn.preprocessing import OneHotEncoder

# ohe = OneHotEncoder(sparse=False)
# # fit_transform은 train에만 사용하고 test에는 학습된 인코더에 fit만 해야한다
# train_cat = ohe.fit_transform(train[['cat1']])
# train_cat
print('y의 라벨값 :', np.unique(y,return_counts=True))
import matplotlib.pylab as plt
# plt.gray()
# plt.matshow(datasets.images[1])
# plt.show()

# print(num)
###########(keras 버전 원핫인코딩)###############
from tensorflow.keras.utils import to_categorical 
y = to_categorical(y) 
print(y.shape) #(1797,10)

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



# print(x.shape, y.shape) #(150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25,shuffle=True ,random_state=100)
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

print(x_train.shape) #(1347, 64)
print(x_test.shape) #(450, 64)

x_train = x_train.reshape(1347, 64,1)
x_test = x_test.reshape(450, 64,1)



#2. 모델 구성

model = Sequential()
# model.add(LSTM(10,input_shape=(64,1)))
model.add(Conv1D(10,2,input_shape=(64,1)))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))
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
from tensorflow.python.keras.callbacks import EarlyStopping,ReduceLROnPlateau

earlyStopping = EarlyStopping(monitor='loss', patience=10, mode='min', 
                              verbose=1,restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=10,
                              mode='auto',verbose=1,factor=0.5)
from tensorflow.python.keras.optimizers import adam_v2
learning_rate = 0.01
optimizer = adam_v2.Adam(lr=learning_rate)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=100, batch_size=1000, 
                validation_split=0.3,
                callbacks = [earlyStopping,reduce_lr],
                verbose=2
                )
#다중 분류 모델은 'categorical_crossentropy'만 사용한다 !!!!

#4.  평가,예측

loss,acc = model.evaluate(x_test,y_test)
print('loss :',loss)

y_predict = model.predict(x_test)
y_test = np.argmax(y_test,axis=1)
print(y_test)

y_predict = np.argmax(y_predict,axis=1)
# y_test와 y_predict의  shape가 일치해야한다.
print(y_predict)

acc = accuracy_score(y_test, y_predict)
print('acc 스코어 :', acc)
end_time =time.time()-start_time
print("걸린 시간 :",end_time)
# drop 아웃 전 
# loss : 1.0825244188308716
# acc 스코어 : 0.4074074074074074
# drop 아웃 후
# loss : 0.12414991110563278
# acc 스코어 : 0.9666666666666667

#cnn dnn 후
# loss : 0.17777284979820251
# acc 스코어 : 0.9666666666666667
######LSTM
# loss : 1.662695288658142
# acc 스코어 : 0.41333333333333333
######LSTM
# loss : 0.1174473986029625
# acc 스코어 : 0.9666666666666667
# 걸린 시간 : 5.282375335693359
######Conv1d + LR reduce

# acc 스코어 : 0.9688888888888889
# 걸린 시간 : 3.609070301055908



