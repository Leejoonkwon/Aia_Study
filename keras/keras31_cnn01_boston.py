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
    x,  y, train_size= 0.7 , random_state=66
)
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
print(x_train.shape) #(354, 13)
print(x_test.shape) #(152, 13)

scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(354, 13,1,1)
x_test = x_test.reshape(152, 13,1,1)
print(np.unique(y_train,return_counts=True))


# print(np.min(x_train)) # 0.0
# print(np.max(x_train)) # 1.0
# print(np.min(x_test)) # -0.06141956477526944  train 범위에서 없는 데이터가 test에 있는 걸 확인할 수 있다.
# print(np.max(x_test)) # 1.1478180091225068
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten

#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(1, 1),   # 출력(4,4,10)                                    
                 padding='same',
                 input_shape=(13, 1, 1)))    #(batch_size, row, column, channels)     
                                                                                           

 #    (kernel_size * channls) * filters = summary Param 개수(CNN모델)  
model.add(Conv2D(32, (1,1),  #인풋쉐이프에 행값은 디폴트는 32
                 padding = 'same',         # 디폴트값(안준것과 같다.) 
                 activation= 'swish'))    # 출력(3,3,7)       
model.add(Conv2D(100, (1,1), 
                 padding = 'same',         # 디폴트값(안준것과 같다.) 
                 activation= 'swish'))    # 출력(3,3,7)      
                                              
model.add(Flatten()) # (N, 63) 위치와 순서는 바뀌지 않아야한다.transpose와 전혀 다르다.
# model.add(Dropout(0.3))
# Dropout 공부해라잉
# 드롭아웃은 신경망 학습 시에만 사용하고, 예측 시에는 사용하지 않는 것이 일반적입니다. 
# 학습 시에 인공 신경망이 특정 뉴런 또는 특정 조합에 너무 의존적이게 되는 것을 방지해주고,
# 매번 랜덤 선택으로 뉴런들을 사용하지 않으므로 서로 다른 신경망들을 앙상블하여 사용하는 것과 
# 같은 효과를 내어 과적합을 방지합니다.
model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(1,activation='softmax'))
model.summary()

import datetime
# #3. 컴파일,훈련
filepath = './_ModelCheckPoint/K24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
#04d :                  4f : 
earlyStopping = EarlyStopping(monitor='loss', patience=10, mode='min', 
                              verbose=1,restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
#                       save_best_only=True, 
#                       filepath="".join([filepath,'k24_', date, '_', filename])
#                     )
model.compile(loss='categorical_crossentropy', optimizer='adam')
# "".join은 " "사이에 있는 문자열을 합치겠다는 기능
hist = model.fit(x_train, y_train, epochs=10, batch_size=15, 
                validation_split=0.2,
                verbose=2,callbacks = [earlyStopping]
                )

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

# print("============")
# print(hist) #<tensorflow.python.keras.callbacks.History object at 0x000001B6B522F0A0>
# print("============")
# # print(hist.history) #(지정된 변수,history)를 통해 딕셔너리 형태의 데이터 확인 가능 
# print("============")
# print(hist.history['loss'])
# print("============")
# print(hist.history['val_loss'])



y_predict = model.predict(x_test)
print(y_test.shape) #(152,)
print(y_predict.shape) #(152, 13, 1)

# y_test= y_test.reshape(152, 13)
# y_predict = y_predict.reshape(152, 13)
# print(y_test.shape) #(152,)
# print(y_predict.shape) #(152, 13, 1)

from sklearn.metrics import accuracy_score, r2_score,accuracy_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)

#################################
# loss : 2.2761423587799072
# r2스코어 : 0.8636196826801059

