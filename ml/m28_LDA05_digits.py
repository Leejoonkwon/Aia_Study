import numpy as np    
from sklearn.datasets import load_breast_cancer,load_wine,load_digits
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import PCA
import sklearn as sk 
import warnings 
warnings.filterwarnings(action='ignore')
# print(sk.__version__) #0.24

#1. 데이터 
datasets =  load_digits()
x = datasets.data
y = datasets.target      
print(x.shape,y.shape) # (1797, 64) (1797,)
# pca = PCA(n_components=50) # 차원 축소 (차원=컬럼,열,피처)
# x = pca.fit_transform(x) 
# print(x.shape) # (506, 2)

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.8,shuffle=True,random_state=123)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis() 
lda.fit(x_train,y_train)
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)

#2. 모델

from xgboost import XGBClassifier,XGBRegressor
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

model = RandomForestRegressor()

#3. 훈련
import time
start_time = time.time()
model.fit(x_train,y_train)
end_time = time.time()

#4. 평가,예측 
results = model.score(x_test,y_test) 
print('model.score :',results)
print("걸린 시간 :",end_time-start_time)
#==========PCA 사용 전
# (569, 30) (569,)
# 결과 : 0.9045683260942199

#==========PCA 사용 후
# model.score : 0.7966313001562625
# 걸린 시간 : 2.0591471195220947

#==========LDA 사용 후
# model.score : 0.8686469793680642
# 걸린 시간 : 0.4156348705291748



