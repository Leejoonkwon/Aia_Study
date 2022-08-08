import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import pandas as pd 
#1.데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
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
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline 

# model = SVC()
# model = make_pipeline(MinMaxScaler(),SVC())
model = make_pipeline(StandardScaler(),RandomForestClassifier())
# 모델 정의와 스케일링을 정의해주지 않아도  fit에서 fit_transform이 적용된다.

#3. 훈련 
model.fit(x_train,y_train)

#4. 평가,예측
result = model.score(x_test,y_test)
# 모델 정의와 상관없이 model로 정의된 기능에 x_test에 transform이 적용된다

print('model.score :',result)

# acc 스코어 : 0.8221842298512942
########### ML 적용 후
# results : 0.505496144666789

