from tabnanny import verbose
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.94,shuffle=True,random_state=12)

# print(x.shape, y.shape) #(506, 13)-> 13개의 피쳐 (506,) 

# print(datasets.feature_names)
# print(datasets.DESCR)


#2. 모델구성
model = Sequential()
model.add(Dense(8, input_dim=13))
model.add(Dense(22))
model.add(Dense(36))
model.add(Dense(28))
model.add(Dense(10))
model.add(Dense(12))
model.add(Dense(14))
model.add(Dense(2))
model.add(Dense(1))
modelimport time
#3. 컴파일,훈련

model.compile(loss='mae', optimizer='adam')

start_time = time.time()
print(start_time)
model.fit(x_train, y_train, epochs=50, batch_size=1, verbose=)
end_time = time.time() - start_time

#verbose = 0으로 할 시 출력해야할 데이터가 없어 속도가 빨라진다.강제 지연 발생을 막는다.

print("걸린시간 :",end_time)

"""
verbose 0 걸린시간 : 15.227273225784302 / 출력 없다.
verbose 1 걸린시간 : 18.025428771972656 / 잔소리많다.
verbose 2 걸린시간 : 14.67668104171753 / 프로그래스바 없다.
verbose 3 걸린시간 : 14.916686058044434 /epoch만 출력 verbose 3이상은 출력내용 동일

"""


