import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#사이킷런 머신러닝 하면서 자주 쓸 예정

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10]) #이런식으로 정렬된 데이터보다 ex)3,4,1,2,9 와 같은 데이터로 하는 게
y = np.array([1,2,3,4,5,6,7,8,9,10]) #안전하다?

# [검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법 찾기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,train_size=0.7, 
                                                #    shuffle=True ,
                                                    random_state=138
)
#셔플의 기본값은 True

print(x_train) # [2 7 6 3 4 8 5]
print(x_test)  # [1 9 10]

print(y_train)
print(y_test)


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(3))
model.add(Dense(1))
model.summary()
'''
#3, 컴파일, 훈련

model.compile(loss='mse', optimizer="adam")
model.fit(x_train, y_train, epochs=300,batch_size=1)

#4. 평가,예측

loss = model.evaluate(x_test, y_test)
print('loss :', loss)
result = model.predict([11])
print('[11]의 예측값 :',result)

#loss : 1.1216201301067485e-06
#[11]의 예측값 : [[10.998204]]

'''