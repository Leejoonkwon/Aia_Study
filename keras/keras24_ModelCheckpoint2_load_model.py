from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
import time

#1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets.target #데이터를 리스트 형태로 불러올 때 함


x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.8,shuffle=True,random_state=100)
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# print(datasets.feature_names)
# print(datasets.DESCR)
'''
#2. 모델구성
model = Sequential()
model.add(Dense(64,input_dim=13))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.summary()



#3. 컴파일,훈련
earlyStopping = EarlyStopping(monitor='loss', patience=50, mode='min', 
                              verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
                      save_best_only=True, 
                      filepath='./_ModelCheckPoint/keras24_ModelCheckPoint.hdf5'
                    )
model.compile(loss='mae', optimizer='adam')

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=50, 
                validation_split=0.2,
                callbacks=[earlyStopping,mcp],
                verbose=2  )
'''
model = load_model('./_ModelCheckpoint/keras24_ModelCheckPoint.hdf5')
# #4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print("r2 :",r2)
###check 사용 전
# loss : 3.0153465270996094
# r2 : 0.74945205417175
###check 사용 후
# loss : 3.0153465270996094
# r2 : 0.74945205417175
