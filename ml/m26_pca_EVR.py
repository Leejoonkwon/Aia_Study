import numpy as np    
from sklearn.datasets import load_boston,fetch_california_housing,load_breast_cancer
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
pca = PCA(n_components=20) # 차원 축소 (차원=컬럼,열,피처)
x = pca.fit_transform(x) 
print(x.shape) # (506, 2)
pca_EVR = pca.explained_variance_ratio_ # PCA로 압축 후에 새로 생성된 피쳐 임포턴스를 보여준다.
print(sum(pca_EVR)) #0.999998352533973
print(pca_EVR)

cumsum = np.cumsum(pca_EVR)
print(cumsum)
import matplotlib.pyplot as plt   
plt.plot(cumsum)
plt.grid()
plt.show() # 그림을 그려서 컬럼이 손실되면 안되는 범위를 예상할 수 있다.
'''
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
# 결과 : 0.7857082622724337

#==========PCA 사용 후
# (506, 11)
# 결과 : 0.7726775210959143
# (506, 12)
# 결과 : 0.7704975850353886
'''