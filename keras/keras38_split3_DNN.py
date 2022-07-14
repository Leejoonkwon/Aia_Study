import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,GRU,LSTM#이름부터 간단?

a = np.array(range(1,101)) #[1,2,3,4,5,6,7,8,9,10]

size = 5 # x= 4개 y는 1개
def split_x(dataset, size): # def라는 예약어로 split_x라는 변수명을 아래에 종속된 기능들을 수행할 수 있도록 정의한다.
    aaa = []   #aaa 는 []라는 값이 없는 리스트임을 정의
    for i in range(len(dataset)- size + 1): # 6이다 range(횟수)
        subset = dataset[i : (i + size)]
        #i는 처음 0에 개념 [0:0+size]
        # 0~(0+size-1인수 까지 )노출 
        aaa.append(subset) #append 마지막에 요소를 추가한다는 뜻
    return np.array(aaa)    


bbb = split_x(a, size)
print(bbb) #(96,5)


x = bbb[:, :-1]
y = bbb[:, -1]

print(x.shape) #(96,4)
print(y.shape) # (96,)
# x = x.reshape(96,4,1)
# y = y.reshape(96,1,1)
# print(x.shape) #(96,4,1)
# print(y.shape) # (96,1,1)
# print(z.shape) # (6, 4)
# print(z) # (6, 4)



#2. 모델구성
model = Sequential()
model.add(Dense(64,input_dim=4,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='relu'))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mae',optimizer='adam')
model.fit(x, y, epochs=100)

#4. 평가, 예측
loss = model.evaluate(x,y)
x_predict = np.array(range(96,106))
# [96,97,98,~~]
ccc = split_x(x_predict, size)
z = ccc[:,:-1]
print(z)#  (6,4)
# [[ 96  97  98  99]
#  [ 97  98  99 100]
#  [ 98  99 100 101]
#  [ 99 100 101 102]
#  [100 101 102 103]
#  [101 102 103 104]]

# q = ccc[:,-1]
# z = z.reshape(6,4,1)
result = model.predict(z)
print('loss ;',loss)
# print(q) # [100 101 102 103 104 105]
print('[96,106]의 결과 :',result)
#############LSTM 결과
# [96,106]의 결과 : 
#  [[100.23987 ]
#  [101.299286]
#  [102.38079 ]
#  [103.43824 ]
#  [104.49248 ]
#  [105.54921 ]]

#############DNN 결과
# 96,106]의 결과 : 
#  [[100.0739  ]
#  [101.07741 ]
#  [102.0809  ]
#  [103.084404]
#  [104.087906]
#  [105.09236 ]]



