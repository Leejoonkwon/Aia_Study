import numpy as np
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,SimpleRNN,Dropout,LSTM#이름부터 간단?
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]
              ,[5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8,shuffle=True ,random_state=100)
print(x_train.shape,x_test.shape) #(10, 3) (3, 3)



print(x_train.shape,x_test.shape) #(10, 3) (3, 3)
x_train = x_train.reshape(10,3,1) 
x_test = x_test.reshape(3,3,1)
'''
#2. 모델구성
model = Sequential()
model.add(LSTM(units=100, input_length=3,input_dim=1,activation='swish')) #위와 같은 개념
model.add(Dense(100,activation='swish')) 
model.add(Dense(100,activation='swish')) 
model.add(Dense(100,activation='swish')) 
model.add(Dense(1))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x, y, epochs=15)

#4. 평가, 예측
loss = model.evaluate(x,y)
y_predict = np.array([50,60,70]).reshape(1,3,1)
print(y_predict)
result = model.predict(y_predict)
print('loss ;',loss)
print('[50,60,70]의 결과 :',result)
#  # 아워너 80
# loss ; 0.016844643279910088
# [50,60,70]의 결과 : [[81.32388]]

# loss ; 0.0998464822769165
# [50,60,70]의 결과 : [[81.79311]]

# loss ; 6.993354320526123
# [50,60,70]의 결과 : [[80.62754]]

# loss ; 5.702090263366699
# [50,60,70]의 결과 : [[79.924484]]
'''