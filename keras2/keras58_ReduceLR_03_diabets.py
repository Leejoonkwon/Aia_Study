#  과제
# activation : sigmoid,relu,linear
# metrics 추가
# EarlyStopping  넣고
# 성능비교
# 감상문 2줄이상!
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D,Flatten,LSTM,Conv1D
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False


#1. 데이터
datasets = load_diabetes()
x = datasets.data #데이터를 리스트 형태로 불러올 때 함
y = datasets.target
x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.91,shuffle=True,random_state=100)
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

# print(x.shape, y.shape) #(442, 10) 442행 10열

# print(datasets.feature_names)
# print(datasets.DESCR)
print(x_train.shape) #(397, 10)
print(x_test.shape) #(45, 10)

x_train = x_train.reshape(402, 10,1)
x_test = x_test.reshape(40, 10,1)

#2. 모델구성
model = Sequential()
# model.add(LSTM(10,input_shape=(10,1)))
model.add(Conv1D(10,2,input_shape=(10,1)))
model.add(Flatten())
model.add(Dense(100,))
# model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(1))
import datetime
date = datetime.datetime.now()
print(date)

date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)


# #3. 컴파일,훈련
filepath = './_ModelCheckPoint/K24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
#04d :                  4f : 
from tensorflow.python.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.python.keras.optimizers import adam_v2
learning_rate = 0.01
optimizer = adam_v2.Adam(lr=learning_rate)
model.compile(loss='mae', optimizer=optimizer,metrics=['mse'])
earlyStopping = EarlyStopping(monitor='loss', patience=10, mode='min', 
                              verbose=1,restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=10,
                              mode='auto',verbose=1,factor=0.5)
import time
start_time = time.time()
print(start_time)
hist = model.fit(x_train, y_train, epochs=310, batch_size=8, 
                validation_split=0.3,
                callbacks = [earlyStopping,reduce_lr],
                verbose=2
                )


#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)
end_time =time.time()-start_time
print("걸린 시간 :",end_time)
# drop 아웃 전 
# loss : 50.59015655517578
# r2스코어 : 0.27348701726188285
# drop 아웃 후
# loss : 41.99888610839844
# r2스코어 : 0.493409441129779
# conv2d
# loss : 49.70637512207031
# r2스코어 : 0.27672623282979936
#######LSTM
# loss : 0.48905089497566223
# r2스코어 : 0.6339364556207625
#######Conv1d
# loss : 53.526763916015625
# r2스코어 : 0.2400053682051626
# 걸린 시간 : 36.60472822189331
#######Conv1d + LR Reduce
# loss : [46.21913146972656, 0.0]
# r2스코어 : 0.4601869160219173
# 걸린 시간 : 38.909528970718384
