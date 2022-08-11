import numpy as np    
from sklearn.datasets import load_iris,load_wine,load_digits
from sklearn.datasets import load_breast_cancer,fetch_covtype
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import PCA
from keras.datasets import mnist 
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from xgboost import XGBClassifier
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#1. 데이터

# datasets = load_iris()               #(150, 4)      - >  (150, 2)
# datasets = load_breast_cancer()      #(569, 30)     - >  (569, 1) 이진분류
# datasets = load_wine()               # (178, 13)    - >  (178, 2)
datasets = fetch_covtype()             # (581012, 54) - >  (581012, 6)
# datasets = load_digits()             # (1797, 64)   - >  (1797, 9)

# LDA는 Y값의 unique를 확인하고 디폴튼 class -1의 값으로 차원을 압축한다.
# ex) (150,4)인 iris의 컬럼은 4개지만 클래스는 3개이므로 디폴트 기준 2개의 컬럼으로 차원을 압축한다.

x = datasets.data
y = datasets.target  
print(np.unique(y, return_counts=True))

'''
print(x.shape)

lda = LinearDiscriminantAnalysis() 
lda.fit(x,y)
x = lda.transform(x)
print(x.shape)

lda_EVR = lda.explained_variance_ratio_ # PCA로 압축 후에 새로 생성된 피쳐 임포턴스를 보여준다.
print(np.cumsum(lda_EVR)) 
'''
