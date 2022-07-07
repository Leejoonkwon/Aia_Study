import numpy as np  
#######################
# 함수형 모델 작성!!!!
# : 함수형 모델은 기존  Sequential  모델과 달리 모델 타입을 레이어 구성 후에 지정해준다.
#######################
#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4],
              [9,8,7,6,5,4,3,2,1,0]]
             )
y = np.array([11,12,13,14,15,16,17,18,19,20])

# x = x.T
# x = x.transtpose(10,3)
x = x.T
print(x.shape) #(10,3)

#2. 모델구성
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense, Input

# model = Sequential()
# # model.add(Dense(10,input_dim=3)) # Total params: 117
# model.add(Dense(10,input_shape=(3,))) 
# model.add(Dense(5,activation='relu'))
# model.add(Dense(3,activation='sigmoid'))
# model.add(Dense(1))
input1 =Input(shape=(3,))
dense1 = Dense(10)(input1)
dense2 = Dense(5,activation='relu')(dense1)
dense3 = Dense(3,activation='sigmoid')(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1)
# 
model.summary()

#3. 컴파일 훈련
model.compile(loss = 'mse',optimizer='adam')
model.fit(x,y ,epochs=10,batch_size=1)

