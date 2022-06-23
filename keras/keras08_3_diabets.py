from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.9,shuffle=True,random_state=10)
# print(x)
# print(y)
# print(x.shape, y.shape) # (442, 10) 442행 10열

# print(datasets.feature_names)
# print(datasets.DESCR)

# [실습]
# R2 0.62 이상

#2. 모델구성
model = Sequential()
model.add(Dense(8, input_dim=10))
model.add(Dense(32))
model.add(Dense(34))
model.add(Dense(38))
model.add(Dense(42))
model.add(Dense(50))
model.add(Dense(32))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일,훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=230, batch_size=18)

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)

# loss : 2155.687744140625
# r2스코어 : 0.6430334416083464