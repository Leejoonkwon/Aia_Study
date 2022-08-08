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
############ ML 시
# results : 0.5387474246360466

