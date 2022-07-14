#12개 만들고 최적의 웨이트 찾기
from sklearn.datasets import load_boston
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler #직접 찾아라!
import numpy as np

datasets = load_boston()
x = datasets.data
y = datasets.target 
#특성 13개

x_train, x_test, y_train, y_test = train_test_split(
    x,  y, train_size= 0.89 , random_state=100
)
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
print(x_train.shape) #(450, 13)
print(x_test.shape) #(56, 13)

scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(450, 13,1)
x_test = x_test.reshape(56, 13,1)


# print(np.min(x_train)) # 0.0
# print(np.min(x_test)) # -0.06141956477526944  train 범위에서 없는 데이터가 test에 있는 걸 확인할 수 있다.
# print(np.max(x_test)) # 1.1478180091225068
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten,LSTM

#2. 모델구성
model = Sequential()
model.add(LSTM(100,input_shape=(13,1)))
model.add(Dense(50, input_dim=13,activation='relu'))
model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.summary()

import datetime
# #3. 컴파일,훈련
filepath = './_ModelCheckPoint/K24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
#04d :                  4f : 
# earlyStopping = EarlyStopping(monitor='loss', patience=10, mode='min', 
#                               verbose=1,restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
#                       save_best_only=True, 
#                       filepath="".join([filepath,'k24_', date, '_', filename])
#                     )
model.compile(loss='mae', optimizer='adam')
# "".join은 " "사이에 있는 문자열을 합치겠다는 기능
hist = model.fit(x_train, y_train, epochs=150, batch_size=30, 
                validation_split=0.2,
                verbose=2,#callbacks = [earlyStopping]
                )

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)


y_predict = model.predict(x_test)
print(y_test.shape) #(152,)
print(y_predict.shape) #(152, 13, 1)

from sklearn.metrics import accuracy_score, r2_score,accuracy_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)

################################# CNN
# loss : 2.2761423587799072
# r2스코어 : 0.8636196826801059
################################# LSTM
# loss : 2.403717517852783
# r2스코어 : 0.8671276057624697
