from unittest import skipUnless
import numpy as  np  
import pandas as pd 
from sklearn.datasets import load_boston,fetch_california_housing
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target
print(x.shape) # (581012, 54)

# pca = PCA(n_components=20)    
# x = pca.fit_transform(x) 
# print(x.shape) # (506, 2)
# pca_EVR = pca.explained_variance_ratio_ # PCA로 압축 후에 새로 생성된 피쳐 임포턴스를 보여준다.
# print(sum(pca_EVR)) #0.999998352533973
# print(pca_EVR)
# cumsum = np.cumsum(pca_EVR)
# print(cumsum)
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(y)


# import matplotlib.pyplot as plt   
# plt.plot(cumsum)
# plt.grid()
# plt.show() # 그림을 그려서 컬럼이 손실되면 안되는 범위를 예상할 수 있다.
# print(np.argmax(cumsum >=0.95)+1)   # 2
# print(np.argmax(cumsum >=0.99)+1)   # 4
# print(np.argmax(cumsum >=0.999)+1)  # 5
# print(np.argmax(cumsum >=1.0)+1) # 52

x_train, x_test,y_train,y_test = train_test_split(x,y,stratify=y,train_size=0.8,random_state=123,shuffle=True)
lda = LinearDiscriminantAnalysis() 
lda.fit(x_train,y_train)
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)
# print(np.unique(y_train,return_counts=True)) # array([1, 2, 3, 4, 5, 6, 7

#2. 모델 구성
from xgboost import XGBClassifier,XGBRegressor
model = XGBRegressor(tree_method='gpu_hist',
                      preditor='gpu_predictor',
                      gpu_id=0)

#3. 훈련
import time
start_time = time.time()
model.fit(x_train,y_train)
end_time = time.time()
#4 . 평가,예측
results = model.score(x_test,y_test)
print('model.score :',results)
print("걸린 시간 :",end_time-start_time)

# xgboost - gpu
# model.score : 0.8949166544753577
# 걸린 시간 : 8.564970970153809

# xgboost - gpu n_component -10
# model.score : 0.8406065247885166
# 걸린 시간 : 4.553236484527588

# xgboost - gpu n_component -20
# model.score : 0.886027038888841
# 걸린 시간 : 5.264762878417969

# xgboost - gpu LDA로
# model.score : 0.8380778441925238
# 걸린 시간 : 0.6942012310028076
