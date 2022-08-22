# 실습
# 증폭한 후 저장
# 피클 or npy로 저장
import numpy as np
from sklearn.datasets import load_iris,load_boston,load_digits,load_breast_cancer
from sklearn.datasets import load_boston,fetch_covtype,fetch_california_housing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel

# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(150, 4) (150,)
import pandas as pd 
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    shuffle=True, random_state=123, 
                                                    train_size=0.88,stratify=y)
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from imblearn.over_sampling import SMOTE
import time
start_time= time.time()
print(pd.Series(y_train).value_counts()) # 데이터 증폭 
smote = SMOTE(random_state=123)

x_train,y_train = smote.fit_resample(x_train,y_train)

print(pd.Series(y_train).value_counts()) # 데이터 증폭 
end_time = time.time()-start_time
print('걸린 시간 :',end_time)

np.save('C:\study\Study\_save\_npy/keras49_9_train_x.npy',arr=x_train)
np.save('C:\study\Study\_save\_npy/keras49_9_train_y.npy',arr=y_train)
np.save('C:\study\Study\_save\_npy/keras49_9_test_x.npy',arr=x_test)
np.save('C:\study\Study\_save\_npy/keras49_9_test_y.npy',arr=y_test)

