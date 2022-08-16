# [실습]
#  아까 4가지 모델을 맹그러봐
# 784개 DNN으로 만든거 (최상의 성능인거)
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
# a = [154,331,486,713]
x_train = x_train.reshape(60000,784)
pca = PCA(n_components=784) # 차원 축소 (차원=컬럼,열,피처)
x_train = pca.fit_transform(x_train) 
print(x_train.shape) # (506, 2)
pca_EVR = pca.explained_variance_ratio_ # PCA로 압축 후에 새로 생성된 피쳐 임포턴스를 보여준다.
print(sum(pca_EVR)) #0.999998352533973
print(pca_EVR)

cumsum = np.cumsum(pca_EVR)
# cumsum = np.argmax(cumsum,axis=1)
print(np.argmax(cumsum >0.9999999)+1) 

x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
# scaler = StandardScaler()
scaler = MinMaxScaler()

scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train_scale = scaler.transform(x_train)
x_test_scale = scaler.transform(x_test)



pca = PCA(n_components=154)
model = XGBClassifier(tree_method='gpu_hist')
# tree_method='gpu_hist' gpu를 사용하겠다는 뜻 데이터가 작을 경우 
# CPU 만으로 하는 것이 빠름 
# predictor = 'gpu_predictor
x_train_pca = pca.fit_transform(x_train_scale) 
x_test_pca = pca.transform(x_test_scale) 

start_time = time.time()
model.fit(x_train_pca,y_train,verbose=1) # eval_metric='error'
end_time = time.time()-start_time
results = model.score(x_test_pca,y_test) 
print('결과 :',results)
print("걸린 시간 :",end_time)
    
# pca = PCA(n_components=331) # 차원 축소 (차원=컬럼,열,피처)


# 0.95 # 154
# 0.99 # 331
# 0.999 # 486
# 1.0 # 713

# #2. 모델

# from xgboost import XGBClassifier,XGBRegressor
# from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

# model = RandomForestClassifier()
# import time
# start_time = time.time()
# #3. 훈련
# model.fit(x_train,y_train) # eval_metric='error'
# end_time = time.time()-start_time
# #4. 평가,예측 
# results = model.score(x_test,y_test) 
# print('결과 :',results)
# print("걸린 시간 :",end_time)

#1. 나의 최고의  DNN
# 걸린 시간 : 7.615624189376831
# acc 스코어 : 0.9645

#2. 나의 최고의  CNN
# 걸린 시간 : 40.151572942733765
# acc 스코어 : 0.9915

#3. PCA 0.95
# 결과 : 0.9501
# 걸린 시간 : 70.41563248634338

#4. PCA 0.99
# 결과 : 0.9417
# 걸린 시간 : 108.65130114555359

#5. PCA 0.999
# 결과 : 0.9700714285714286
# 걸린 시간 : 29.213647842407227

#6. PCA 1.0
# 결과 : 0.9255
# 걸린 시간 : 157.61064887046814


