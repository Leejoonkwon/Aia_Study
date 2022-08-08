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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor #공부하자 
from sklearn.ensemble import RandomForestRegressor #공부하자 
from sklearn.linear_model import LogisticRegression 

#1. 데이터
datasets = load_boston()
x = datasets.data #데이터를 리스트 형태로 불러올 때 함
y = datasets.target
from sklearn.svm import LinearSVC 
from sklearn.svm import LinearSVR 
from sklearn.preprocessing import MinMaxScaler,StandardScaler

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.8
       ,random_state=1234,shuffle=True)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train) 
# x_test = scaler.transform(x_test)

#2. 모델 
from sklearn.svm import LinearSVC, SVC 
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline 

# model = SVC()
# model = make_pipeline(MinMaxScaler(),SVC())
model = make_pipeline(StandardScaler(),RandomForestRegressor())
# 모델 정의와 스케일링을 정의해주지 않아도  fit에서 fit_transform이 적용된다.

#3. 훈련 
model.fit(x_train,y_train)

#4. 평가,예측
result = model.score(x_test,y_test)
# 모델 정의와 상관없이 model로 정의된 기능에 x_test에 transform이 적용된다

print('model.score :',result)
###################### ML LinearSVR
# results : 0.6975307894843537
# results : 0.5554437069519558
