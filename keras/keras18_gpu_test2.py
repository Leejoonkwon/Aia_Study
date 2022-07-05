from time import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if(gpus) : 
    print("쥐피유 돈다")
    aaa = 'gpu'
else:
    print("쥐피유 안도라")
    bbb = 'cpu'


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
print(x_test.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(500, activation='linear',input_dim=30))
model.add(Dense(400, activation='relu'))
model.add(Dense(300, activation='relu')) #'relu'가 현시점 가장 성능 좋음
model.add(Dense(400, activation='relu')) #'relu'가 현시점 가장 성능 좋음
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
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='loss', patience=200, mode='min', 
                              verbose=1,restore_best_weights=True)
#monitor -확인할 대상+fit 안에서만 기능하는 / patience- 최솟값 이후 멈추기 전 횟수 /mode- 스탑 결정할 모델 
import time
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=200, batch_size=15, 
                validation_split=0.3,
                callbacks = [earlyStopping],
                verbose=2
                )
#callbacks 함수는 earlystopping 외에도 checkpoint가 있다.

end_time = time.time()-start_time

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

print(aaa,"걸린시간 :",end_time)



# GPU 환경에서 걸린시간 : 25.983750820159912
# GPU 환경에서 걸린시간 : 12.76516604423523

