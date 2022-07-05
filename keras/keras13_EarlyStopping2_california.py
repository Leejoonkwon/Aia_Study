from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data #데이터를 리스트 형태로 불러올 때 함
y = datasets.target
x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.91,shuffle=True,random_state=68)

# print(x.shape, y.shape) 
# print(datasets.feature_names)
# print(datasets.DESCR)


#2. 모델구성
model = Sequential()
model.add(Dense(100,input_dim=8))
model.add(Dense(100, activation='swish'))
model.add(Dense(100, activation='swish'))
model.add(Dense(1))

#3. 컴파일,훈련

model.compile(loss='mae', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='loss', patience=100, mode='min', 
                              verbose=1,restore_best_weights=True)
#monitor -확인할 대상+fit 안에서만 기능하는 / patience- 최솟값 이후 멈추기 전 횟수 /mode- 스탑 결정할 모델 
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=150, 
                validation_split=0.25,
                verbose=2,callbacks = [earlyStopping]
                )

end_time = time.time() - start_time

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
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)

print("걸린시간 :",end_time)
# y_predict = model.predict(x_test)
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'],marker='.',c='red',label='loss') #순차적으로 출력이므로  y값 지정 필요 x
plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss')
plt.grid()
plt.title('영어싫어') #맥플러립 한글 깨짐 현상 알아서 해결해라 
plt.ylabel('loss')
plt.xlabel('epochs')
# plt.legend(loc='upper right')
plt.legend()
plt.show()
# validation 적용 전
# [실습 시작!!] #0.54이상
# loss : 0.5898192524909973
# r2스코어 : 0.5617319420274233
#################
# validation 적용 후
# loss : 0.5187357664108276
# r2스코어 : 0.606785090759131
##################
# EarlyStopping 적용 후
# loss : 0.4959072172641754
# r2스코어 : 0.08726959581939397


