from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.94,shuffle=True, random_state=12 ) 
print(x)
print(y)
print(x.shape, y.shape) #(1797, 64) (1797,)

print(datasets.feature_names)
print(datasets.DESCR)

#2. 모델 구성

model = Sequential()
model.add(Dense(73,input_dim=64))
model.add(Dense(83))
model.add(Dense(32))
model.add(Dense(28))
model.add(Dense(20))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 150, batch_size = 12)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 스코어 :', r2)

