# csv로 맹그러!!!
from tkinter import Y
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import StandardScaler,MinMaxScaler 
from xgboost import XGBClassifier,XGBRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#1. 데이터
path = 'C:\_data\wine/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'winequality-white.csv',
                        sep=';',index_col=None,header=0)
print(train_set.describe())

x = train_set.drop(['quality'],axis=1)
y = train_set['quality']
print(np.unique(y,return_counts=True))

print(x.shape,y.shape) # (4898, 11) (4898,)
newlist = []
for i in y :
    if i<=5:
        newlist += [0]
    elif i<=6:
        newlist += [1]
    else:
        newlist += [2]    
# print(np.unique(newlist,return_counts=True))
# (array([0, 1, 2]), array([1640, 2198, 1060], dtype=int64))       
# y['band'] = pd.cut(y, 3)
# # 임의로 5개 그룹을 지정
# print(y['band'])
# [(2.994, 4.2] < (4.2, 5.4] < (5.4, 6.6] <
#  (6.6, 7.8] <   (7.8, 9.0]]
# [(2.994, 5.0] < (5.0, 7.0] < (7.0, 9.0]]

 

# print(np.unique(y,return_counts=True))
# (array([3, 4, 5, 6, 7, 8, 9], dtype=int64), 
#  array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
# pandas 타입이라면 df['확인할 컬럼'].value_counts()로 확인 가능 
# from tensorflow.keras.utils import to_categorical 
# y = to_categorical(y)
# print(x.shape)
# print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    shuffle=True, random_state=123, 
                                                    train_size=0.8,stratify=y)
from sklearn.preprocessing import MinMaxScaler,StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)

score = model.score(x_test,y_test)
from sklearn.metrics import accuracy_score,f1_score
print('model.score :', score)
print('acc_score :',accuracy_score(y_test,y_predict))
# print('f1_score(macro) :',f1_score(y_test,y_predict,average='macro'))
# f1_score(macro) : 0.4397558777039733 이진 분류일 때 사용
print('f1_score(micro) :',f1_score(y_test,y_predict,average='micro'))
# f1_score(micro) : 0.7163265306122448 다중 분류일 때 사용


# 과제 f1 스코어 

