
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC 
from sklearn.svm import LinearSVR 
from sklearn.metrics import r2_score


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data #데이터를 리스트 형태로 불러올 때 함
y = datasets.target
from sklearn.model_selection import KFold,cross_val_score,cross_val_predict
import warnings 
warnings.filterwarnings('ignore')
x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.8,shuffle=True,random_state=100)
n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)
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

#2. 모델 구성
from sklearn.utils import all_estimators

allAlgorithms = all_estimators(type_filter='regressor')
# allAlgorithms = all_estimators(type_filter='Regressor')

# print('allAlgorithms :',allAlgorithms)
print('모델의 갯수 :',len(allAlgorithms)) #모델의 갯수 : 41

for (name,algorithms) in allAlgorithms:
    try: # for문을 실행하는 와중에 예외 (error)가 발생하면 무시하고 진행 <예외처리>
        model = algorithms()
        model.fit(x_train,y_train)
        
        y_predict = model.predict(x_test)
        # acc = accuracy_score(y_test,y_predict)
        r2  = r2_score(y_test,y_predict)
        scores = cross_val_score(model, x, y, cv=kfold)
        # print('{}의 정확도 {}의 ',name,'의 정답률 :',acc)
        print('{}의 r2_score :{} 검증 평균: {} '.format(name,round(r2,3),round(np.mean(scores),4)))
       
    except:
        continue 