import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC 
from sklearn.svm import LinearSVR 
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
#1. 데이터
datasets= load_breast_cancer()
x = datasets['data']
y = datasets['target']

x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.8,shuffle=True,random_state=100)
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_test.shape)

#2. 모델구성
model = LinearSVC()
#3. 컴파일,훈련

model.fit(x_train,y_train)

#4. 평가,예측
results = model.score(x_test,y_test) #분류 모델과 회귀 모델에서 score를 쓰면 알아서 값이 나온다 
#ex)분류는 ACC 회귀는 R2스코어
print("results :",results)
# print("============")
# print(hist) #<tensorflow.python.keras.callbacks.History object at 0x000001B6B522F0A0>
# print("============")
# # print(hist.history) #(지정된 변수,history)를 통해 딕셔너리 형태의 데이터 확인 가능 
# print("============")
# print(hist.history['loss'])
# print("============")


#cc 스코어 : acc 스코어 : 0.9130434782608695
##################
#1. 스케일러 하기전
# loss : 
# acc 스코어 : :0.9130434782608695
##################
#2. 민맥스
# loss :  [0.08987700939178467, 0.9736841917037964, 0.024136152118444443]
# acc 스코어 : : 0.9736842105263158
##################
#3. 스탠다드
# loss : [0.0943220779299736, 0.9473684430122375, 0.029835190623998642]
# acc 스코어 : 0.9473684210526315
# 걸린시간 : 2.292578935623169
#4. 절댓값
# loss : [0.136466383934021, 0.9298245906829834, 0.04213083162903786]
# acc 스코어 : 0.9298245614035088
# 걸린시간 : 2.2992329597473145
#5. RobustScaler
# loss : [0.09985142201185226, 0.9561403393745422, 0.03045959398150444]
# acc 스코어 : 0.956140350877193
# 걸린시간 : 2.373518228530884

#######ML 사용시
# results : 0.9649122807017544
