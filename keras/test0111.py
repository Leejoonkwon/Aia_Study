from sklearn import metrics
from sklearn.datasets import load_boston

data_sets = load_boston()

x = data_sets.data
y = data_sets.target

print(x.shape, y.shape)

#sklearn dataset만 먹는 명령어
print(data_sets.DESCR)

# Number of Instances: 506

#     :Number of Attributes: 13 

#x,y 는 이미 설정돼있음 모델 만들면 됨 train test 셋 나눠줘야함

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=False)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv1D,LSTM,MaxPool1D,Flatten
x_train=x_train.reshape(-1,13,1)
x_test=x_test.reshape(-1,13,1)
print(x.shape)
model = Sequential()
model.add(Conv1D(5,2, input_shape=(13,1)))
model.add(MaxPool1D(10))
model.add(Flatten())
model.add(Dense(5, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(5))
model.add(Dense(1))
model.summary()

#컴파일 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics=['accuracy', 'mae'])
#컴파일 전후 얼리스타핑 미니멈 혹은 맥시멈값을 patience 지켜보고 있다가 정지시키는 함수

from tensorflow.python.keras.callbacks import EarlyStopping
ES = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=2, restore_best_weights=True)

#훈련을 인스턴스 하나로 줄여주기
h = model.fit(x_train, y_train, epochs=10, batch_size=10,
          validation_split=0.1, 
          callbacks=[ES], 
          verbose=3)

#훈련돼ㅆㅇ니 평가 예측

loss = model.evaluate(x_test, y_test)
print('loss', loss)

print(h) #훈련 h가 어느 메모리에 저장돼있는가
print('=======================================')
print(h.history) #훈련 h의 loss율 그리고 추가돼있다면 validation loss 율\
  
  
  
print(y_test)
print(x_test,x_test.shape) #(102, 13, 1) #(102, 13, 1)
y_predict = model.predict(x_test)
print(y_predict,y_predict.shape) #(102, 1, 1) #(102, 1)
# print(y_train.shape)
# print(y_test.shape)

# y_predict = y_predict.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
# print(y_predict.shape)
# print(y_test.shape)
