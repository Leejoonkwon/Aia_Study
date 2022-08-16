# 실습!! 시작 !!!
# csv로 맹그러!!!
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

x_train, x_test, y_train, y_test = train_test_split(x, newlist, 
                                                    shuffle=True, random_state=123, 
                                                    train_size=0.89,stratify=newlist)
print(pd.Series(y_train).value_counts()) # 데이터 증폭 
# 1    1758
# 0    1312
# 2     848
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=123,k_neighbors=3)
x_train,y_train = smote.fit_resample(x_train,y_train)

print(pd.Series(y_train).value_counts()) # 데이터 증폭 
# 1    1758
# 0    1758
# 2    1758

#2. 모델
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)

score = model.score(x_test,y_test)
from sklearn.metrics import accuracy_score,f1_score
print('acc_score :',accuracy_score(y_test,y_predict))
print('f1_score(macro) :',f1_score(y_test,y_predict,average='macro'))
# f1_score(macro) : 0.4397558777039733 이진 분류일 때 사용
# print('f1_score(micro) :',f1_score(y_test,y_predict,average='micro'))
# f1_score(micro) : 0.7163265306122448 다중 분류일 때 사용

######################## SMOTE 만 적용시 
# acc_score : 0.6802721088435374
# f1_score(macro) : 0.42765507578677014

######################## Label 축소 후 SMOTE까지
# acc_score : 0.7402597402597403
# f1_score(macro) : 0.7432630229600647
