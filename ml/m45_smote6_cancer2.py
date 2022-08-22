#라벨 1 357개
# 0 212개

# 라벨 0을 112개 삭제해서 재구성

#smote 너서 맹그러
# 는거 안는거 비교 
# 지표는 ACC,F1 매크로 아니고 기본값으로 
# smote 너서 맹그러봐
# 있고 없고
# csv로 맹그러!!!
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import StandardScaler,MinMaxScaler 
from xgboost import XGBClassifier,XGBRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data

x = x[112:]
y = datasets.target
y = np.sort(y)
y = y[112:]

# index = np.argwhere(y<1)

print(y)
# print(x.shape,y.shape) #(569, 30) (569,)

print(pd.Series(y).value_counts()) 
# 데이터 증폭 
# 1    357
# 0    212

print(x.shape,y.shape) # (4898, 11) (4898,)



from imblearn.over_sampling import SMOTE

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    shuffle=True, random_state=123, 
                                                    train_size=0.88,stratify=y)
from sklearn.preprocessing import MinMaxScaler,StandardScaler
print(pd.Series(y_train).value_counts()) # 데이터 증폭 
smote = SMOTE(random_state=123)
x_train,y_train = smote.fit_resample(x_train,y_train)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(pd.Series(y_train).value_counts()) # 데이터 증폭 

#2. 모델
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)

score = model.score(x_test,y_test)
from sklearn.metrics import accuracy_score,f1_score
# print('model.score :', score)
print('acc_score :',accuracy_score(y_test,y_predict))
# print('f1_score(macro) :',f1_score(y_test,y_predict,average='macro'))
# f1_score(macro) : 0.4397558777039733 이진 분류일 때 사용
print('f1_score(micro) :',f1_score(y_test,y_predict,average='micro'))
# f1_score(micro) : 0.7163265306122448 다중 분류일 때 사용
######################## SMOTE 미적용
# acc_score : 0.9710144927536232
# f1_score(micro) : 0.97101449275362

######################## SMOTE 적용
# acc_score : 0.9710144927536232
# f1_score(micro) : 0.9710144927536232

# acc_score : 0.9818181818181818
# f1_score(micro) : 0.9818181818181818


