import numpy as np  
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

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
model = Sequential()
# model.add(Dense(10,input_dim=3)) # (100,3) -> (None, 3)  행은 중요하지 않다.
# # Total params: 117
model.add(Dense(10,input_shape=(3,))) #Total params: 117 input_shape에서 (3,0)은 스칼라 3개 벡터 1개라는 뜻 이미지를 인풋하기 
# 위해서 shape로 넣는다.예시로 이미지 파일이 가로 28 /세로 28 /흑백에 /6만장 있다면 shape는 (28,28,1)로 표현한다.
# 6만장은 행이므로 무시한다.
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))
model.summary()

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape
#      Param #
# =================================================================
# dense (Dense)                (None, 10)
#      40
# _________________________________________________________________
# dense_1 (Dense)              (None, 5)
#      55
# _________________________________________________________________
# dense_2 (Dense)              (None, 3)
#      18
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)
#      4
# =================================================================
# Total params: 117
# Trainable params: 117
# Non-trainable params: 0



