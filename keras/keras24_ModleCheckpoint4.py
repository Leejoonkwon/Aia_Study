from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
import time

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target #데이터를 리스트 형태로 불러올 때 함


x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.8,shuffle=True,random_state=100)
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(datasets.feature_names)
# print(datasets.DESCR)

#2. 모델구성
model = Sequential()
model.add(Dense(64,input_dim=13))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.summary()

import datetime
date = datetime.datetime.now()
print(date)

date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)


# #3. 컴파일,훈련
filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
#04d :                  4f : 
earlyStopping = EarlyStopping(monitor='loss', patience=10, mode='min', 
                              verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
                      save_best_only=True, 
                      filepath="".join([filepath,'k24_', date, '_', filename])
                    )
# "".join은 " "사이에 있는 문자열을 합치겠다는 기능
model.compile(loss='mae', optimizer='adam')


model.fit(x_train, y_train, epochs=100, batch_size=50, 
                validation_split=0.2,
                callbacks=[earlyStopping,mcp],
                verbose=2  )


# # #4. 평가,예측
print("=========================1.기본출력========================")
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print("r2 :",r2)
'''
print("=========================2.load_model 출력========================")
model2 = load_model('./_save/keras24_3_save_model.h5')
loss2 = model2.evaluate(x_test, y_test)
print('loss :', loss2)

y_predict2 = model2.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict2)
print("r2 :",r2)

print("=========================3.ModelCheckpoint 출력========================")
model3 = load_model('./_ModelCheckPoint/keras24_ModelCheckpoint3.hdf5')
loss3 = model3.evaluate(x_test, y_test)
print('loss :', loss3)

y_predict3 = model3.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict3)
print("r2 :",r2)

# =========================1.기본출력========================
# 1/4 [======>.......................] - ETA: 0s - l4/4 [==============================] - 0s 2ms/step - loss: 9.4026
# loss : 9.402571678161621
# r2 : -0.5285233444774788
# =========================2.load_model 출력========================
# 1/4 [======>.......................] - ETA: 0s - l4/4 [==============================] - 0s 2ms/step - loss: 9.4026
# loss : 9.402571678161621
# r2 : -0.5285233444774788
# =========================3.ModelCheckpoint 출력========================
# 1/4 [======>.......................] - ETA: 0s - l4/4 [==============================] - 0s 3ms/step - loss: 9.4026
# loss : 9.402571678161621
# r2 : -0.5285233444774788
'''