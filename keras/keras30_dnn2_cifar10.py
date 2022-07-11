from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout 
from keras.datasets import mnist,cifar10
import pandas as pd
import numpy as np


#1. 데이터 전처리

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape,y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape,y_test.shape) #(10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(50000, 32*32* 3)
x_test = x_test.reshape(10000, 32*32* 3)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape) #(50000, 32, 32, 3, 1)
print(x_test.shape) #(10000, 32, 32, 3, 1)
print(np.unique(y_train,return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
#        dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
#       dtype=int64))
print(y_train.shape) #(50000, 1)
print(y_test.shape) #(10000, 1)
from tensorflow.keras.utils import to_categorical 
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test) 

# y_train = pd.get_dummies((y_train)) 
# y_test = pd.get_dummies((y_test))
print(y_train.shape) #(50000, 10)
print(y_test.shape) #(10000, 10)


#2. 모델 구성
model = Sequential()
# model.add(Flatten()) #  해도 돌아감
model.add(Dense(1000,input_shape=(3072,),activation='swish'))
model.add(Dropout(0.3))
model.add(Dense(1000,activation='swish'))
model.add(Dropout(0.3))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='sigmoid'))
model.summary()

#3. 컴파일 훈련
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
earlyStopping = EarlyStopping(monitor='loss', patience=150, mode='min', 
                              verbose=1,restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1000, batch_size=4500, 
                callbacks = [earlyStopping],
                verbose=2
                )
#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score,accuracy_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)
y_test = np.argmax(y_test,axis=1)
print(y_test)

y_predict = np.argmax(y_predict,axis=1)
# y_test와 y_predict의  shape가 일치해야한다.
print(y_predict)


acc = accuracy_score(y_test, y_predict)
print('acc 스코어 :', acc)

# StandardScaler 시
# loss : [2.9135234355926514, 0.6779000163078308]
# r2스코어 : -1.844786723836507
# acc 스코어 : 0.6908

# MinMaxScaler 시
# loss : [4.197653770446777, 0.6668000221252441]
# r2스코어 : -1.7121904232828808
# acc 스코어 : 0.5816

# MaxAbsScaler 시
# loss : [4.466394424438477, 0.6656000018119812]
# r2스코어 : -1.8811170396280275
# acc 스코어 : 0.5721

#conv2d 포함
# loss : [2.9135234355926514, 0.6779000163078308]
# r2스코어 : -1.844786723836507
# acc 스코어 : 0.6908


#conv2d 미포함 reshape 바로 Dense
# loss : [2.2696797847747803, 0.590399980545044]
# r2스코어 : -4.116410039075688
# acc 스코어 : 0.5442