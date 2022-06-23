from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7,shuffle=True, random_state=12 ) 
# print(x)
# print(y)
print(x.shape, y.shape) #(150, 4) (150,)

print(datasets.feature_names)
print(datasets.DESCR)

#2. 모델 구성

model = Sequential()
model.add(Dense(3,input_dim=4))
model.add(Dense(2))
model.add(Dense(2))
model.add(Dense(2))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 100, batch_size = 2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 스코어 :', r2)

# loss : 0.03997482359409332
# r2 스코어 : 0.9383949644473354