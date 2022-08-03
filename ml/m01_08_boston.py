#  과제

# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.rcParams['font.family']='Malgun Gothic'
# matplotlib.rcParams['axes.unicode_minus']=False
import time
from sklearn.svm import LinearSVC 
from sklearn.svm import LinearSVR 


#1. 데이터
datasets = load_boston()
x = datasets.data #데이터를 리스트 형태로 불러올 때 함
y = datasets.target
x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.75,shuffle=True,random_state=100)


#2. 모델구성

model = LinearSVR()
# #3. 컴파일,훈련

model.fit(x_train,y_train)



#4. 평가,예측

results = model.score(x_test,y_test) #분류 모델과 회귀 모델에서 score를 쓰면 알아서 값이 나온다 
print("results :",results)


# validation 적용 전
# r2 스코어 "0.8이상"
# loss : 3.1941933631896973
# r2스코어 : 0.8146167932510104
#################
# validation 적용 후
# loss : 2.74114727973938
# r2스코어 : 0.8827299842129269
#################
# EarlyStopping 및 activation 적용 버전
# loss : 3.5245087146759033
# r2스코어 : 0.7557914895748932
###################### ML LinearSVR
# results : 0.6975307894843537
# results : 0.5554437069519558
