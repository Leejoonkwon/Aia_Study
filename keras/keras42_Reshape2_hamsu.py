from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout
from tensorflow.python.keras.layers import LSTM,Conv1D,Reshape,Input
from keras.datasets import mnist,cifar10,cifar100
import pandas as pd
import numpy as np


#1. 데이터 전처리
# datasets = mnist.load_data()
# x = datasets.data #데이터를 리스트 형태로 불러올 때 함
# y = datasets.target
# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25,shuffle=True ,random_state=100)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape,y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape,y_test.shape) #(60000,) (10000,)

# x_train = x_train.reshape(60000, 28*28*1)
# x_test = x_test.reshape(10000, 28*28*1)
x_train = x_train.reshape(60000, 28*28*1)
x_test = x_test.reshape(10000, 28*28*1)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = StandardScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(60000, 28,28,1)
x_test = x_test.reshape(10000, 28,28,1)
print(np.unique(y_train,return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
#  array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#       dtype=int64))

y_train = pd.get_dummies((y_train)) 
y_test = pd.get_dummies((y_test))



#2. 모델 구성
model = Sequential()

# model.add(Conv2D(filters=64, kernel_size=(3, 3),                                     
#                  padding='same',
#                  input_shape=(28, 28, 1)))                                                                                               
# model.add(MaxPooling2D())                         # (N,14,14,64)   
# model.add(Conv2D(32, (3,3), activation= 'relu'))  # (N,12,12,32)         
# model.add(Conv2D(7, (3,3), activation= 'relu'))   # (N,10,10,7)   
# model.add(Flatten())                              # (N, 700)
# model.add(Dense(100,activation='relu'))           # (N, 100)
# model.add(Reshape(target_shape=(100,1)))          # (N, 100,1)
# model.add(Conv1D(10,3))                           # (N, 100,10) 실제 #(N ,98,10)  커널사이즈 반영 안함 
# # model.add(LSTM(16))                               # (N, 16)
# model.add(Reshape(target_shape=(98,10,1)))         # (N, 100,1)
# model.add(Conv2D(10,kernel_size=(1,1)))                           # (N, 100,10) 실제 #(N ,98,10)  커널사이즈 반영 안함 
# model.add(Reshape(target_shape=(9800,1)))          # (N, 100,1)
# # model.add(Flatten())
# # model.add(LSTM(16))                               # (N, 16)
# model.add(Dense(32, activation='relu'))           # (N, 32)
# model.add(Dense(10, activation='sigmoid'))        # (N, 10)
input1 =Input(shape=(28,28,1))                      # (N,28,28,1)
dense1 = Conv2D(10,(2,2))(input1)                    # (N,27,27,10)
dense2 = Reshape(target_shape=(7290,1))(dense1)       # (N,7290,1)
dense3 = LSTM(10)(dense2)                             # (N,10)
dense4 = Dense(3,activation='sigmoid')(dense3)        # (N,3)
output1 = Dense(10)(dense4)                          # (N,10)
model = Model(inputs=input1, outputs=output1)
model.summary()

'''
import datetime
date = datetime.datetime.now()
print(date)

date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)

#3. 컴파일 훈련
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
filepath = './_ModelCheckPoint/K24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
earlyStopping = EarlyStopping(monitor='loss', patience=5, mode='min', 
                              verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
                      save_best_only=True, 
                      filepath="".join([filepath,'k25_', date, '_mnist2_', filename])
                    )
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100000, batch_size=3500, 
                validation_split=0.33,
                callbacks = [earlyStopping,mcp],
                verbose=2
                )

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score,accuracy_score
# print(y_test.shape)

# y_test = np.argmax(y_test,axis=1)
print(y_predict)

y_predict = np.argmax(y_predict,axis=1)
# y_test와 y_predict의  shape가 일치해야한다.
print(y_test) #[10000 rows x 10 columns]
y_predict = pd.get_dummies((y_predict))


acc = accuracy_score(y_test, y_predict)
print('acc 스코어 :', acc)

# loss : [0.03594522178173065, 0.9915000200271606]
# acc 스코어 : 0.9915
'''