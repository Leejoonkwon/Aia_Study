
import numpy as np  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
    
#1. 데이터
x = np.array([{'aa' : [1,2,3,4,5,6,7,8,9,10],
              'bb':[1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4]}]
             )
y = np.array([1,1,5,8,9,5,2,1,4,1,3,6,9,5,5,4,8])
from tensorflow.keras.utils import to_categorical 
print(x)

print('y의 라벨값 :', np.unique(y,return_counts=True))
y = to_categorical(y) 
print(y)

y['bb'] = x['bb']

print(y)

print(x.shape) # (2, 10)
print(y.shape) # (10,)

# x = x.T
# x = x.transtpose(10,2)
x = x.T
print(x)
print(x.shape) #(10,2)


#2. 모델구성
model = Sequential()
model.add(Dense(5,input_dim=2))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y , epochs=800, batch_size=1)

#4. 평가,예측
loss = model.evaluate(x,y)
print('loss :',loss)
result = model.predict([[10, 1.4]])
print('[10,1.4]의 예측값 :',result)

#loss : 4.490359970077407e-06
#[10,1.4]의 예측값 : [[19.996462]]
df = pd.DataFrame({'hour': [hour], 'hour_bef_temperature': [hour_bef_temperature], 'hour_bef_precipitation': [hour_bef_precipitation],
       'hour_bef_windspeed': [hour_bef_windspeed], 'hour_bef_humidity': [hour_bef_humidity], 'hour_bef_visibility': [hour_bef_visibility],
       'hour_bef_ozone': [hour_bef_ozone], 'hour_bef_pm10': [hour_bef_pm10], 'hour_bef_pm2.5': [hour_bef_pm25]})
df.fillna(method='pad')
