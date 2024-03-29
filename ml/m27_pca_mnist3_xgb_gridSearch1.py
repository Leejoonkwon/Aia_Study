# n_component > 0.95 이상 
# xgboost,gridSearch 또는 RandomSearch를 쓸것

# m27_2 결과를 뛰어넘어랏
import numpy as np    
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import PCA
from keras.datasets import mnist 
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from xgboost import XGBClassifier
import time

(x_train, y_train), (x_test, y_test)=mnist.load_data()  
# print(y_train.shape)
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
# scaler = StandardScaler() 
scaler = MinMaxScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train_scale = scaler.transform(x_train)
x_test_scale = scaler.transform(x_test)
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,StratifiedKFold
# a = [154,331,486,713]

pca = PCA(n_components=154)
# tree_method='gpu_hist' gpu를 사용하겠다는 뜻 데이터가 작을 경우 
# CPU 만으로 하는 것이 빠름 
# predictor = 'gpu_predictor
x_train_pca = pca.fit_transform(x_train_scale) 
x_test_pca = pca.transform(x_test_scale) 

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=100)

parameters = [
    {"n_estimators":[100,200,300],"learning_rate":[0.1, 0.3, 0.001,0.01],
     "max_depth": [4,5,6]},
    {"n_estimators":[90,100,110],"learning_rate":[0.1, 0.001, 0.01],
     "max_depth": [4,5,6]},
    {"n_estimators":[90,110],"learning_rate":[0.1, 0.001, 0.5],
     "max_depth": [4,5,6],"colsample_bytree":[0.6,0.9,1],
     "colsample_bylevel":[0.6,0.7,0.9]}
]

model = GridSearchCV(XGBClassifier(tree_method='gpu_hist'),parameters,cv=kfold,verbose=1,
                     refit=True,n_jobs=-1) 


start_time = time.time()
model.fit(x_train_pca,y_train,verbose=1) # eval_metric='error'
end_time = time.time()-start_time
results = model.score(x_test_pca,y_test) 
print('결과 :',results)
print("걸린 시간 :",end_time)
    

#  GridSearchCV 154












