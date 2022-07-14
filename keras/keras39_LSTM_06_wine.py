import numpy as np
from sklearn.datasets import load_wine
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten,LSTM
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.metrics import accuracy_score
#1.데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(178, 13) (178,)

print(np.unique(y,return_counts=True))
print(datasets.DESCR)
print(datasets.feature_names)
x = datasets.data
y = datasets['target']
# from sklearn.preprocessing import OneHotEncoder

# ohe = OneHotEncoder(sparse=False)
# # fit_transform은 train에만 사용하고 test에는 학습된 인코더에 fit만 해야한다
# train_cat = ohe.fit_transform(train[['cat1']])
# train_cat
print('y의 라벨값 :', np.unique(y))

# print(num)
###########(keras 버전 원핫인코딩)###############
from tensorflow.keras.utils import to_categorical 
y = to_categorical(y) 

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



print(x.shape, y.shape) #178, 13) (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.15,shuffle=True ,random_state=100)
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
# scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
# scaler.transform(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
#셔플을 False 할 경우 순차적으로 스플릿하다보니 훈련에서는 나오지 않는 값이 생겨 정확도가 떨어진다.
#디폴트 값인  shuffle=True 를 통해 정확도를 올린다.
print(y_train,y_test)

print(x_train.shape) #(151, 13)
print(x_test.shape) #27, 13)

x_train = x_train.reshape(151, 13,1)
x_test = x_test.reshape(27, 13,1)



#2. 모델 구성

model = Sequential()
model.add(LSTM(10,input_shape=(13,1)))
model.add(Dense(100))
# model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))
#다중 분류로 나오는 아웃풋 노드의 개수는 y 값의 클래스의 수와 같다.활성화함수 'softmax'를 통해 
# 아웃풋의 합은 1이 된다.
import datetime
date = datetime.datetime.now()
print(date)

date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)


# #3. 컴파일,훈련
filepath = './_ModelCheckPoint/K24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
#04d :                  4f : 
earlyStopping = EarlyStopping(monitor='loss', patience=10, mode='min', 
                              verbose=1,restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
#                       save_best_only=True, 
#                       filepath="".join([filepath,'k25_', date, '_wine_', filename])
#                     )
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=250, batch_size=15, 
                validation_split=0.3,
                callbacks = [earlyStopping],
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

# drop 아웃 전 
# loss : 0.37419381737709045
# acc 스코어 : 0.9259259259259259
# drop 아웃 후
# loss : 0.3554897904396057
# r2스코어 : 0.7876406039148734
#cnn dnn 후
# loss : 0.2589413821697235
# acc 스코어 : 0.8888888888888888
# ######LSTM
# loss : 0.1397644728422165
# acc 스코어 : 0.9259259259259259