#  과제
# activation : sigmoid,relu,linear
# metrics 추가
# EarlyStopping  넣고
# 성능비교
# 감상문 2줄이상!
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten,LSTM,Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
import pandas as pd
from sklearn.svm import LinearSVC 
from sklearn.svm import LinearSVR 
#1. 데이터
path = 'D:\study_data\_data\_csv\_ddarung/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
print(train_set)

print(train_set.shape) #(1459, 10)

test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)
submission = pd.read_csv(path + 'submission.csv',#예측에서 쓸거야!!
                       index_col=0)
                       
print(test_set)
print(test_set.shape) #(715, 9) #train_set과 열 값이 '1'차이 나는 건 count를 제외했기 때문이다.예측 단계에서 값을 대입

print(train_set.columns)
print(train_set.info()) #null은 누락된 값이라고 하고 "결측치"라고도 한다.
print(train_set.describe()) 

###### 결측치 처리 1.제거##### dropna 사용
print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
train_set = train_set.fillna(train_set.median())
print(train_set.isnull().sum())
print(train_set.shape)
test_set = test_set.fillna(test_set.median())

x = train_set.drop(['count'],axis=1) #axis는 컬럼 
print(x.columns)
print(x.shape) #(1459, 9)

y = train_set['count']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.919, shuffle = True, random_state = 100)
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(y)

from tqdm import tqdm
from sklearn.svm import LinearSVC,SVC,SVR
from sklearn.linear_model import Perceptron ,LogisticRegression 
#LogisticRegression은 유일하게 Regression이름이지만 분류 모델이다.

#LogisticRegression은 유일하게 Regression이름이지만 분류 모델이다.
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor #공부하자 
from sklearn.ensemble import RandomForestRegressor #공부하자 
from sklearn.linear_model import LinearRegression 
def models(model):
    if model == 'knn':
        mod = KNeighborsRegressor()
    elif model == 'svr':
        mod = SVR()
    elif model == 'tree':
        mod =  DecisionTreeRegressor()
    elif model == 'forest':
        mod =  RandomForestRegressor()
    elif model == 'linear':
        mod =  LinearRegression ()    
    return mod
model_list = ['knn', 'svr',  'tree', 'forest','linear']
empty_list = [] #empty list for progress bar in tqdm library
for model in (model_list):
    empty_list.append(model) # fill empty_list to fill progress bar
    #classifier
    clf = models(model)
    #Training
    clf.fit(x_train, y_train) 
    #Predict
    result = clf.score(x_test,y_test)
    pred = clf.predict(x_test) 
    print('{}-{}'.format(model,result))
    

# knn-0.7528726805652388
# svr-0.5035232735894759
# tree-0.6940573133343038
# forest-0.7986739602497022
# linear-0.6069571332489507
