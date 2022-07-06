#  과제
# activation : sigmoid,relu,linear
# metrics 추가
# EarlyStopping  넣고
# 성능비교
# 감상문 2줄이상!
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
import time

#1. 데이터
datasets = load_boston()
x = datasets.data #데이터를 리스트 형태로 불러올 때 함
y = datasets.target
x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.91,shuffle=True,random_state=100)

# print(x.shape, y.shape) #(506, 13)-> 13개의 피쳐 (506,) 

# print(datasets.feature_names)
# print(datasets.DESCR)


#2. 모델구성
model = Sequential()
model.add(Dense(100,input_dim=13))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.summary()
# Total params: 21,701
# Trainable params: 21,701
# Non-trainable params: 0
'''
#3. 컴파일,훈련
earlyStopping = EarlyStopping(monitor='loss', patience=100, mode='min', 
                              verbose=2,restore_best_weights=True)
model.compile(loss='mae', optimizer='adam')

start_time = time.time()
print(start_time)
hist = model.fit(x_train, y_train, epochs=1010, batch_size=15, 
                validation_split=0.3,
                callbacks = [earlyStopping],
                verbose=2
                )

end_time = time.time() - start_time

#verbose = 0으로 할 시 출력해야할 데이터가 없어 속도가 빨라진다.강제 지연 발생을 막는다.



#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)

# print("============")
# print(hist) #<tensorflow.python.keras.callbacks.History object at 0x000001B6B522F0A0>
# print("============")
# # print(hist.history) #(지정된 변수,history)를 통해 딕셔너리 형태의 데이터 확인 가능 
# print("============")
# print(hist.history['loss'])
# print("============")
# print(hist.history['val_loss'])
# print("걸린시간 :",end_time)
# # y_predict = model.predict(x_test)
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
# r2 스코어 "0.8이상"
# loss : 3.1941933631896973
# r2스코어 : 0.8146167932510104
#################
# validation 적용 후
# loss : 2.74114727973938
# r2스코어 : 0.8827299842129269
#################
# EarlyStopping 및 activation 적용 버전
# loss : 3.5245087146759033
# r2스코어 : 0.7557914895748932
'''
