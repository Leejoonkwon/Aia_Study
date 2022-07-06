# [과제] 맹그러서 속도 비교
# gpu 와 cpu
import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import pandas as pd 
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if(gpus) : 
    print("쥐피유 돈다")
    aaa = 'gpu'
else:
    print("쥐피유 안도라")
    bbb = 'cpu'

#1.데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(581012, 54) (581012,)

# from sklearn.preprocessing import OneHotEncoder
print('y의 라벨값 :', np.unique(y,return_counts=True))
# y의 라벨값 : (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
    #   dtype=int64))


###########(pandas 버전 원핫인코딩)###############
y_class = pd.get_dummies((y))
print(y_class.shape) # (581012, 7)

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



# print(x.shape, y.shape) #(581012, 54) (581012,)

x_train, x_test, y_train, y_test = train_test_split(x,y_class, test_size=0.15,shuffle=True ,random_state=100)
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
# scaler.transform(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
#셔플을 False 할 경우 순차적으로 스플릿하다보니 훈련에서는 나오지 않는 값이 생겨 정확도가 떨어진다.
#디폴트 값인  shuffle=True 를 통해 정확도를 올린다.
print(y_train,y_test)




#2. 모델 구성

model = Sequential()
model.add(Dense(500,input_dim=54))
model.add(Dense(400, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(7, activation='softmax'))
#다중 분류로 나오는 아웃풋 노드의 개수는 y 값의 클래스의 수와 같다.활성화함수 'softmax'를 통해 
# 아웃풋의 합은 1이 된다.

import time
start_time = time.time()
#3. 컴파일,훈련
earlyStopping = EarlyStopping(monitor='loss', patience=250, mode='min', 
                              verbose=1,restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=14000, 
                validation_split=0.3,
                callbacks = [earlyStopping],
                verbose=2
                )
#다중 분류 모델은 'categorical_crossentropy'만 사용한다 !!!!

#4.  평가,예측

# loss,acc = model.evaluate(x_test,y_test)
# print('loss :',loss)
# print('accuracy :',acc)
# print("+++++++++  y_test       +++++++++")
# print(y_test[:5])
# print("+++++++++  y_pred     +++++++++++++")
# result = model.evaluate(x_test,y_test) 위에와 같은 개념 [0] 또는 [1]을 통해 출력가능
# print('loss :',result[0])
# print('accuracy :',result[1])




y_predict = model.predict(x_test)

print(y_predict) 

# y_test = np.argmax(y_test,axis=1)
import tensorflow as tf
y_test = tf.argmax(y_test,axis=1)
y_predict = tf.argmax(y_predict,axis=1)
#pandas 에서 인코딩 진행시 argmax는 tensorflow 에서 임포트한다.
print(y_predict) #(87152, )
# print(y_test.shape) #(87152,7)
# y_test와 y_predict의  shape가 일치해야한다.



acc = accuracy_score(y_test, y_predict)
print('acc 스코어 :', acc)
end_time = time.time()-start_time
print(aaa,"걸린시간 :",end_time)

# gpu 걸린시간 : 72.09812474250793
# cpu 걸린시간 : 165.26057076454163

##################
#1. 스케일러 하기전
# acc 스코어 : 0.5015145951900128
##################
#2. 민맥스
# acc 스코어 : 0.7641476959794382
##################
#3. 스탠다드
# acc 스코어 : 0.8464751239214247



