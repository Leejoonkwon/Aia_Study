from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time

#1. 데이터
datasets = load_boston()
x = datasets.data #데이터를 리스트 형태로 불러올 때 함
y = datasets.target

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
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.summary()


# model.save("./_save/keras23_1_save.model.h5")
# model.save_weights("./_save/keras23_5_save_weights1.h5")

# model = load_model("./_save/keras23_5_save_weights1.h5")
model.load_weights("./_save/keras23_5_save_weights1.h5")

model.load_weights("./_save/keras23_5_save_weights2.h5")

#3. 컴파일,훈련
earlyStopping = EarlyStopping(monitor='loss', patience=10, mode='min', 
                              verbose=1,restore_best_weights=True)
model.compile(loss='mae', optimizer='adam')

start_time = time.time()
print(start_time)
# model.fit(x_train, y_train, epochs=100, batch_size=50, 
#                 validation_split=0.2,
#                 callbacks=[earlyStopping],
#                 verbose=2, 
#                 )



# model.save("./_save/keras23_3_save.model.h5")
# model.save_weights("./_save/keras23_5_save_weights2.h5")

# model = load_model("./_save/keras23_3_save.model.h5")

# #4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print("r2 :",r2)
# loss : 2.718148946762085
# r2 : 0.7745354687411947

# loss : 2.718148946762085
# r2 : 0.7745354687411947

