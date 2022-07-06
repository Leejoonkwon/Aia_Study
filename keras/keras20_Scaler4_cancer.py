import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
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
#sigmoid는 0~1 사이의 값으로 반환하는 활성화함수 이진 분류 모델에서 아웃풋 활성화함수는 무조건  sigmoid 사용
#회귀 모델은 최종 노드가 무조건 linear인 디폴트 사용 하지만 분류 모델에서는 
#이진 분류시 아웃풋에 sigmoid 사용 loss 모델은 반드시 binary_crossentropy사용

#3. 컴파일,훈련

model.compile(loss='binary_crossentropy', optimizer='adam', 
             metrics=['accuracy','mse'])
# 평가 지표를 loss 하나에서 metrics까지 추가했다.
#'binary_cross_entropy'(이진 교차 엔트로피 손실) 분류 모델이 2가지일 경우 사용
# 크로스 엔트로피 손실(Cross Entropy Loss)은 머신 러닝의 분류 모델이 얼마나
# 잘 수행되는지 측정하기 위해 사용되는 지표입니다. Loss(또는 Error)는0은 완벽한 모델로
# 0과 1 사이의 숫자로 측정됩니다. 일반적인 목표는 모델을 가능한 0에 가깝게 만드는 것입니다.
# 반면Binary Cross Entropy Loss는 하나의 값만 저장합니다. 
# 즉, 0.5만 저장하고 다른 0.5는 다른 문제에서 가정하며, 첫 번째 확률이 0.7이면 다른 0.5는 0.3)이라고 가정합니다. 
# 또한 알고리즘(Log loss)을 사용합니다.
import time
start_time = time.time()
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='loss', patience=100, mode='min', 
                              verbose=1,restore_best_weights=True)
#monitor -확인할 대상+fit 안에서만 기능하는 / patience- 최솟값 이후 멈추기 전 횟수 /mode- 스탑 결정할 모델 
hist = model.fit(x_train, y_train, epochs=10, batch_size=15, 
                validation_split=0.3,
                callbacks = [earlyStopping],
                verbose=2
                )
#callbacks 함수는 earlystopping 외에도 checkpoint가 있다.

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
print(x_test.shape)

y_predict[(y_predict<0.5)] = 0  
y_predict[(y_predict>=0.5)] = 1  

print(y_predict)
print(y_predict.shape)


# r2 = r2_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)
print('acc 스코어 :', acc)
end_time = time.time() - start_time
print("걸린시간 :",end_time)

# y_predict = model.predict(x_test)
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'],marker='.',c='red',label='loss') #순차적으로 출력이므로  y값 지정 필요 x
# plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss')
# plt.grid()
# plt.title('영어싫어') #맥플러립 한글 깨짐 현상 알아서 해결해라 
# plt.ylabel('loss')
# plt.xlabel('epochs')
# # plt.legend(loc='upper right')
# plt.legend()
# plt.show()
#cc 스코어 : acc 스코어 : 0.9130434782608695
##################
#1. 스케일러 하기전
# loss : 
# acc 스코어 : :0.9130434782608695
##################
#2. 민맥스
# loss :  [0.08987700939178467, 0.9736841917037964, 0.024136152118444443]
# acc 스코어 : : 0.9736842105263158
##################
#3. 스탠다드
# loss : [0.0943220779299736, 0.9473684430122375, 0.029835190623998642]
# acc 스코어 : 0.9473684210526315
# 걸린시간 : 2.292578935623169
#4. 절댓값
# loss : [0.136466383934021, 0.9298245906829834, 0.04213083162903786]
# acc 스코어 : 0.9298245614035088
# 걸린시간 : 2.2992329597473145
#5. RobustScaler
# loss : [0.09985142201185226, 0.9561403393745422, 0.03045959398150444]
# acc 스코어 : 0.956140350877193
# 걸린시간 : 2.373518228530884