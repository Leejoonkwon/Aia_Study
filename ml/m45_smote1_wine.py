import numpy as np
import pandas as pd 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score,f1_score
from imblearn.over_sampling import SMOTE
# imblearnd은 cmd에서 pip 인스톨해야 사용 가능 
import sklearn as sk 
print('사이킷런 :',sk.__version__) #사이킷런 : 1.1.2
# 사이킷런이 업데이트 되었으니 문제 생길 가능성 있다.

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']
# print(x.shape,y.shape)  # (178, 13) (178,)
# print(type(x))          # <class 'numpy.ndarray'>
print(np.unique(y,return_counts=True))
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64)
print(pd.Series(y).value_counts())
# 1    71
# 0    59
# 2    48
# dtype: int64
print(y)
x_new = x[:-30]
y_new = y[:-30]
print(pd.Series(y_new).value_counts())

x_train,x_test,y_train,y_test = train_test_split(x_new,y_new,train_size=0.8,shuffle=True,
                                                 random_state=123,stratify=y_new)

print(pd.Series(y_train).value_counts())

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


