# 데이콘 따릉이 문제풀이
import numpy as np
import pandas as pd
from sqlalchemy import null
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = './_data/ddarung/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
print(train_set)

print(train_set.shape) #(1459, 10)

test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)
print(test_set)
print(test_set.shape) #(715, 9) #train_set과 열 값이 '1'차이 나는 건 count를 제외했기 때문이다.예측 단계에서 값을 대입

print(train_set.columns)
print(train_set.info()) #null은 누락된 값이라고 하고 "결측치"라고도 한다.
print(train_set.describe()) 

###### 결측치 처리 1.제거##### dropna 사용
print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
train_set = train_set.fillna(train_set.mean())
print(train_set.isnull().sum())
print(train_set.shape)


x = train_set.drop(['count'],axis=1) #axis는 컬럼 
print(x.columns)
print(x.shape) #(1459, 9)

y = train_set['count']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.9, shuffle = True, random_state = 16
 )
print(y)
print(y.shape) # (1459,)


#2. 모델구성

model = Sequential()
model.add(Dense(10,input_dim=9))
model.add(Dense(96))
model.add(Dense(182))
model.add(Dense(23))
model.add(Dense(28))
model.add(Dense(84))
model.add(Dense(48))
model.add(Dense(12))
model.add(Dense(18))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y , epochs = 2060, batch_size=150, verbose=3)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :',loss)

y_predict = model.predict(x_test) #훈련으로 예측된 값 y_predict와 원래 테스트 값 y_test와 비교

def RMSE(y_test, y_predict):
     return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE :",rmse)  
  
# loss : 2022.3809814453125
# RMSE : 44.97089119001261