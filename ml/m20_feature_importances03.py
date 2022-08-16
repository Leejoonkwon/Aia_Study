###   [실습] 02번을 가져와서 피처 한개 삭제하고 성능 비교 

import numpy as np  
from sklearn.datasets import load_diabetes    
import pandas as pd

#1. 데이터
datasets = load_diabetes()
x =  np.array(datasets.data)
x = np.delete(x, 1 , axis = 1)
# print(arr.shape)
# print(datasets['feature_names'])



y =  datasets.target   
# column_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# data_df = pd.DataFrame(x1,columns=column_names)
# print(data_df)
# data_df = data_df.drop(['age'],axis=1)
# print(data_df)


from sklearn.model_selection import train_test_split 

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.8,shuffle=True,random_state=123)


#2. 모델 구성
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor 
from xgboost import XGBClassifier,XGBRegressor 

# model = DecisionTreeRegressor() 
model = RandomForestRegressor() 
# model  = GradientBoostingRegressor()
# model  = XGBRegressor()
#3. 훈련 
model.fit(x_train,y_train)

#4. 평가,예측

result = model.score(x_test,y_test)
print("model.score :",result)

from sklearn.metrics import accuracy_score, r2_score 
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print('r2_score :',r2)


print("=============================")
print(model,':',model.feature_importances_)

# feature_importtances는 y값에 영향을 미치는 정도를 수치화한다.
# iris는 컬럼이 4개지만 데이터가 더 방대해질경우 필요한 것과 불필요한 것을 분리할 수 있다.
# 여러가지 모델을 비교해서 영향을 확인해야한다.
#===============컬럼 제거 전 (가장 작은 인수 1)===========================================
# DecisionTreeRegressor() : [0.10435529 0.02171698 0.22898906 0.05458676 0.03858524     ㅣ
#                            0.0525853  0.03863638 0.02242778 0.36608123 0.07203599]    ㅣ
# model.score : 0.16460218742421662                                                     ㅣ
# r2_score : 0.16460218742421662                                                        ㅣ
# ===============컬럼 제거 후============================================================
# DecisionTreeRegressor() : [0.10239823 0.02503308 0.22975945 0.05437489 0.043256   
#                            0.04355809 0.03866493 0.02525257 0.3636572  0.07404557] 
# model.score : 0.14715660140607834
# r2_score : 0.14715660140607834
# ===============컬럼 제거 전 (가장 작은 인수 1)==========================================
# RandomForestRegressor() : [0.05645885 0.00990777 0.27930196 0.10773298 0.0406001  
#                            0.05511041 0.05479674 0.02652711 0.27875095 0.09081314] 
# model.score : 0.5285162928080618
# r2_score : 0.5285162928080618
#===============컬럼 제거 후=============================================================
# RandomForestRegressor() : [0.05909148 0.01277897 0.29582334 0.10153286 0.03787086 
#                            0.04844197 0.06017459 0.02822932 0.27770804 0.07834858]
# model.score : 0.5211984765555915
# r2_score : 0.5211984765555915
#===============컬럼 제거 전 (가장 작은 인수 1)============================================
# GradientBoostingRegressor() : [0.04983822 0.01079916 0.30199132 0.11189149 0.02805566 
#                                0.0581757  0.04065599 0.01636338 0.33859948 0.04362959]
# model.score : 0.5575872699933667
# r2_score : 0.5575872699933667
#===============컬럼 제거 후==============================================================
# GradientBoostingRegressor() : [0.04954464 0.01098092 0.302226   0.11170755 0.02903104 
#                                0.05706824 0.04077182 0.01615467 0.33830866 0.04420646]
# model.score : 0.5584186350292104
# r2_score : 0.5584186350292104
#===============컬럼 제거 전 (가장 작은 인수 1)
# XGBRegressor() : [0.03234756 0.0447546  0.21775807 0.08212128 0.04737141 
#                   0.04843819 0.06012432 0.09595273 0.30483875 0.06629313]
# model.score : 0.4590400803596264
# r2_score : 0.4590400803596264
#===============컬럼 제거 후
# XGBRegressor() : [0.03234756 0.0447546  0.21775807 0.08212128 0.04737141 
#                   0.04843819 0.06012432 0.09595273 0.30483875 0.06629313]
# model.score : 0.4590400803596264
# r2_score : 0.4590400803596264
