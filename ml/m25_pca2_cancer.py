import numpy as np    
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import PCA
import sklearn as sk 
import warnings 
warnings.filterwarnings(action='ignore')
# print(sk.__version__) #0.24

#1. 데이터 
datasets =  load_breast_cancer()
x = datasets.data
y = datasets.target      
# print(x.shape,y.shape) # (506, 13) (506,)
pca = PCA(n_components=4) # 차원 축소 (차원=컬럼,열,피처)
x = pca.fit_transform(x) 
print(x.shape) # (506, 2)

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.8,shuffle=True,random_state=123)

#2. 모델

from xgboost import XGBClassifier,XGBRegressor
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

model = RandomForestRegressor()

#3. 훈련
model.fit(x_train,y_train,) # eval_metric='error'

#4. 평가,예측 
results = model.score(x_test,y_test) 
print('결과 :',results)

#==========PCA 사용 전
# (569, 30) (569,)
# 결과 : 0.9045683260942199
#==========PCA 사용 후
# (569, 4)
# 결과 : 0.9143914467089876
# (569, 5)
# 결과 : 0.8971448045439359
# (569, 14)
# 결과 : 0.8997272302038088