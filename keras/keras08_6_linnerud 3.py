from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_linnerud
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_linnerud()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.9, shuffle = True, random_state = 12
 )
print(x)
print(y)
print(x.shape, y.shape) #(20, 3) (20, 3)

print(datasets.feature_names)
print(datasets.DESCR)
print(datasets.target_names)

# #2. 모델구성
# model = Sequential()
# model.add(Dense(4,input_dim=3))
# model.add(Dense(5))
# model.add(Dense(6))
# model.add(Dense(8))
# model.add(Dense(6))
# model.add(Dense(4))
# model.add(Dense(4))
# model.add(Dense(3))

# #3. 컴파일,훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x_train, y_train, epochs=800,batch_size=1)

# #4. 평가,예측
# loss = model.evaluate(x_test, y_test)
# print('loss :',loss)

# y_predict= model.predict(x_test)
# r2 = r2_score(y_test,y_predict)
# print('r2 :', r2)



