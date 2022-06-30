from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.9,shuffle=True,random_state=100)
# print(x)
# print(y)
# print(x.shape, y.shape) # (442, 10) 442행 10열

# print(datasets.feature_names)
# print(datasets.DESCR)

# [실습]
# R2 0.62 이상

#2. 모델구성
model = Sequential()
model.add(Dense(100,input_dim=10))
model.add(Dense(100, activation='swish'))
model.add(Dense(100, activation='swish'))
model.add(Dense(1))

#3. 컴파일,훈련

model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1,
          validation_split=0.25,verbose=2)

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)

# validation 적용 전
# loss : 2155.687744140625
# r2스코어 : 0.6430334416083464
#################
# validation 적용 후(+mae로 측정)
# loss : 39.017478942871094
# r2스코어 : 0.5731992742645576



