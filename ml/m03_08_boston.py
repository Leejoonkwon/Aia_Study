#2. 모델 구성
##### 분류 모데ㅐㄹ 
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression 

#LogisticRegression은 유일하게 Regression이름이지만 분류 모델이다.
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor #공부하자 
from sklearn.ensemble import RandomForestRegressor #공부하자 
from sklearn.linear_model import LogisticRegression 
model = LinearSVC() # DL과 다르게 단층 레이어  구성으로 연산에 걸리는 시간을 비교할 수 없다.
