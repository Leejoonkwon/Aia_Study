from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.94,shuffle=True,random_state=12)

# print(x.shape, y.shape) #(506, 13)-> 13개의 피쳐 (506,) 

# print(datasets.feature_names)
# print(datasets.DESCR)

# [실습] 아래를 완성할 것 
# 1. train 0.7
#2. R2 0.8 이상

#2. 모델구성
model = Sequential()
model.add(Dense(8, input_dim=13))
model.add(Dense(22))
model.add(Dense(36))
model.add(Dense(28))
model.add(Dense(10))
model.add(Dense(12))
model.add(Dense(14))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일,훈련

model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=150, batch_size=2)

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)

# r2 스코어 "0.8이상"
# loss : 3.1941933631896973
# r2스코어 : 0.8146167932510104
