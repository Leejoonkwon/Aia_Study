from tabnanny import verbose
import numpy as np    
from sklearn.datasets import load_breast_cancer,load_wine,fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn as sk 
import warnings 
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings(action='ignore')
# print(sk.__version__) #0.24

#1. 데이터 
datasets =  fetch_covtype()
x = datasets.data
y = datasets.target      
# print(x.shape,y.shape) # (506, 13) (506,)
le = LabelEncoder()
y = le.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,stratify = y,
                                                 train_size=0.8,shuffle=True,random_state=123)
lda = LinearDiscriminantAnalysis() 
lda.fit(x_train,y_train)
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)

#2. 모델

from xgboost import XGBClassifier,XGBRegressor
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

model = XGBClassifier(tree_method='gpu_hist')

#3. 훈련
model.fit(x_train,y_train) # eval_metric='error'

#4. 평가,예측 
results = model.score(x_test,y_test) 
print('결과 :',results)

#==========PCA 사용 전
# (569, 30) (569,)
# 결과 : 0.9045683260942199
#==========PCA 사용 후
# DecisionTreeClassifier 의 스코어:         0.9444444444444444
# DecisionTreeClassifier 의 드랍후 스코어:  0.8333333333333334     
# RandomForestClassifier 의 스코어:         0.9629629629629629
# RandomForestClassifier 의 드랍후 스코어:  0.9629629629629629
# GradientBoostingClassifier 의 스코어:         0.8888888888888888
# GradientBoostingClassifier 의 드랍후 스코어:  0.8888888888888888
# XGB 의 스코어:         0.9259259259259259
# XGB 의 드랍후 스코어:  0.9444444444444444
#==========LDA 사용 후
# 결과 : 0.936923076923077



