import numpy as np
from sklearn.datasets import load_wine
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
#1.데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']
# print(x.shape, y.shape) #(178, 13) (178,)

# print(datasets.DESCR)
# print(datasets.feature_names)
x = datasets.data
y = datasets['target']
from sklearn.svm import LinearSVC 
from sklearn.svm import LinearSVR 



print(x.shape, y.shape) #178, 13) (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.15,shuffle=True ,random_state=100)
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
# scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
# scaler.transform(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
#셔플을 False 할 경우 순차적으로 스플릿하다보니 훈련에서는 나오지 않는 값이 생겨 정확도가 떨어진다.
#디폴트 값인  shuffle=True 를 통해 정확도를 올린다.
print(y_train,y_test)

#2. 모델구성
model = LinearSVC()
#3. 컴파일,훈련

model.fit(x_train,y_train)

#4. 평가,예측
results = model.score(x_test,y_test) #분류 모델과 회귀 모델에서 score를 쓰면 알아서 값이 나온다 
#ex)분류는 ACC 회귀는 R2스코어
print("results :",results)


# acc = accuracy_score(y_test, y_predict)
# print('acc 스코어 :', acc)
# end_time = time.time() - start_time
# print("걸린시간 :",end_time)




##################
#1. 스케일러 하기전
# acc 스코어 : 0.8888888888888888
##################
#2. MinMaxScaler
# loss : 0.608286440372467
# acc 스코어 : 0.9259259259259259
# 걸린시간 : 9.761307954788208
##################
#3. StandardScaler
# loss : 0.19748736917972565
# acc 스코어 : 0.9259259259259259
# 걸린시간 : 10.098722696304321
##################
#4. MaxAbsScaler
# loss : 0.4304962754249573
# acc 스코어 : 0.8148148148148148
# 걸린시간 : 9.927936792373657
##################
#5. RobustScaler
# loss : 0.43310320377349854
# acc 스코어 : 0.8518518518518519
# 걸린시간 : 9.608951568603516

################ ML
# results : 1.0
