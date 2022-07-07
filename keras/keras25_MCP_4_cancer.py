import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
#1. 데이터
datasets= load_breast_cancer()


# print(datasets)
# print(datasets.DESCR) #(569,30)

# print(datasets.feature_names)

x = datasets['data']

y = datasets['target']

print(x.shape,y.shape) #(569, 30) (569,)

print(y)

x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.8,shuffle=True,random_state=100)
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_test.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(100, activation='linear',input_dim=30))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='relu')) #'relu'가 현시점 가장 성능 좋음
model.add(Dense(1,activation='sigmoid'))
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
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
                      save_best_only=True, 
                      filepath="".join([filepath,'k25_', date, '_cancer_', filename])
                    )
model.compile(loss='binary_crossentropy', optimizer='adam', 
             metrics=['accuracy','mse'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='loss', patience=100, mode='min', 
                              verbose=1,restore_best_weights=True)
#monitor -확인할 대상+fit 안에서만 기능하는 / patience- 최솟값 이후 멈추기 전 횟수 /mode- 스탑 결정할 모델 
model.fit(x_train, y_train, epochs=10, batch_size=15, 
                validation_split=0.3,
                callbacks = [earlyStopping,mcp],
                verbose=2
                )

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)


y_predict = model.predict(x_test)
print(x_test.shape)

y_predict[(y_predict<0.5)] = 0  
y_predict[(y_predict>=0.5)] = 1  

print(y_predict)
print(y_predict.shape)


# r2 = r2_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)
print('acc 스코어 :', acc)
