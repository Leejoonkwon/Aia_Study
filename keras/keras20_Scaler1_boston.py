from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
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

scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train)) # 0.0
print(np.max(x_train)) # 1.0
print(np.min(x_test)) # -0.06141956477526944  train 범위에서 없는 데이터가 test에 있는 걸 확인할 수 있다.
print(np.max(x_test)) # 1.1478180091225068

#2. 모델구성
model = Sequential()
model.add(Dense(100,input_dim=13))
model.add(Dense(100, activation='swish'))
model.add(Dense(100, activation='swish'))
model.add(Dense(1))

#3. 컴파일,훈련

model.compile(loss='mae', optimizer='adam')
import time
start_time = time.time()
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='loss', patience=100, mode='min', 
                              verbose=1,restore_best_weights=True)
#monitor -확인할 대상+fit 안에서만 기능하는 / patience- 최솟값 이후 멈추기 전 횟수 /mode- 스탑 결정할 모델 
hist = model.fit(x_train, y_train, epochs=300, batch_size=15, 
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
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)
end_time = time.time() - start_time
print("걸린 시간 :", end_time)

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

#################################

# 보스턴에 대해 3가지 비교
#1. 스케일러 하기전
# loss : 3.1941933631896973
# r2스코어 : 0.8146167932510104

#2. 민맥스
# loss : 2.4100306034088135
# r2스코어 : 0.8553779164226989
# 걸린 시간 : 24.65588116645813
#3. 스탠다드
# loss : 2.596243143081665
# r2스코어 : 0.8408439029524983
# 걸린 시간 : 24.737785577774048
#4. 절댓값
# loss : 2.8734331130981445
# r2스코어 : 0.7695440276622324
# 걸린 시간 : 24.929853677749634
#5. RobustScaler
# loss : 2.805102586746216
# r2스코어 : 0.6344604436586637
# 걸린 시간 : 25.684573888778687
