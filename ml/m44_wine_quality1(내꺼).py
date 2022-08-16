# csv로 맹그러!!!
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import StandardScaler,MinMaxScaler 
from xgboost import XGBClassifier,XGBRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#1. 데이터
path = 'C:\_data\wine/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'winequality-white.csv',
                        sep=';',index_col=None,header=0)
print(train_set.describe())

# print(train_set.isnull().sum()) #(4898, 12)
# def outliers(data_out):
#     quartile_1, q2 , quartile_3 = np.percentile(data_out,
#                                                [25,50,75]) # percentile 백분위
#     print("1사분위 : ",quartile_1) # 25% 위치인수를 기점으로 사이에 값을 구함
#     print("q2 : ",q2) # 50% median과 동일 
#     print("3사분위 : ",quartile_3) # 75% 위치인수를 기점으로 사이에 값을 구함
#     iqr =quartile_3-quartile_1  # 75% -25%
#     print("iqr :" ,iqr)
#     lower_bound = quartile_1 - (iqr * 1.5)
#     upper_bound = quartile_3 + (iqr * 1.5)
#     return np.where((data_out>upper_bound)|
#                     (data_out<lower_bound))
# fixed_acidity_out_index= outliers(train_set['fixed acidity'])[0]
# volatile_acidity_out_index= outliers(train_set['volatile acidity'])[0]
# citric_acid_out_index= outliers(train_set['citric acid'])[0]
# residual_sugar_out_index= outliers(train_set['residual sugar'])[0]
# chlorides_out_index= outliers(train_set['chlorides'])[0]
# free_sulfur_dioxide_out_index= outliers(train_set['free sulfur dioxide'])[0]
# total_sulfur_dioxide_out_index= outliers(train_set['total sulfur dioxide'])[0]
# density_out_index= outliers(train_set['density'])[0]
# pH_out_index= outliers(train_set['pH'])[0]
# sulphates_out_index= outliers(train_set['sulphates'])[0]
# alcohol_out_index= outliers(train_set['alcohol'])[0]

# lead_outlier_index = np.concatenate((fixed_acidity_out_index,
#                                      volatile_acidity_out_index,
#                                      citric_acid_out_index,
#                                      residual_sugar_out_index,
#                                      chlorides_out_index,
#                                      free_sulfur_dioxide_out_index,
#                                      total_sulfur_dioxide_out_index,
#                                      density_out_index,
#                                      pH_out_index,
#                                      sulphates_out_index,
#                                      sulphates_out_index,
#                                      alcohol_out_index,
                                   
#                                      ),axis=None)
# print(len(lead_outlier_index)) #577
# # print(lead_outlier_index)

# lead_not_outlier_index = []
# for i in train_set.index:
#     if i not in lead_outlier_index :
#         lead_not_outlier_index.append(i)
# train_set_clean = train_set.loc[lead_not_outlier_index]      
# train_set_clean = train_set_clean.reset_index(drop=True)

##############Pandas Dataframe을 Numpy로 바꾸기 
# .values 또는 .to_numpy() 를 사용해 numpy 배열로 변환
# train_set=train_set.to_numpy()
train_set=train_set.values
# print(type(train_set)) #<class 'numpy.ndarray'>

# x = train_set.drop(['quality'],axis=1)
# y = train_set['quality']
x = train_set[:,:11]
y = train_set[:,11]
print(x.shape,y.shape) # (4898, 11) (4898,)

# print(np.unique(y,return_counts=True))
# (array([3, 4, 5, 6, 7, 8, 9], dtype=int64), 
#  array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
# pandas 타입이라면 df['확인할 컬럼'].value_counts()로 확인 가능 
# from tensorflow.keras.utils import to_categorical 
# y = to_categorical(y)
# print(x.shape)
# print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    shuffle=True, random_state=123, 
                                                    train_size=0.8,stratify=y)
from sklearn.preprocessing import MinMaxScaler,StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)

score = model.score(x_test,y_test)
from sklearn.metrics import accuracy_score,f1_score
print('model.score :', score)
print('acc_score :',accuracy_score(y_test,y_predict))
# print('f1_score(macro) :',f1_score(y_test,y_predict,average='macro'))
# f1_score(macro) : 0.4397558777039733 이진 분류일 때 사용
print('f1_score(micro) :',f1_score(y_test,y_predict,average='micro'))
# f1_score(micro) : 0.7163265306122448 다중 분류일 때 사용


# 과제 f1 스코어 
'''
n_splits = 7
kfold = KFold(n_splits=n_splits,shuffle=True,random_state=123)
# parameters = {'n_estimators':[100], # 디폴트 100/ 1~inf 무한대 
# eta[기본값=0.3, 별칭: learning_rate]
# gamma[기본값=0, 별칭: min_split_loss]
# max_depth[기본값=6]
# min_child_weight[기본값=1] 0~inf
# subsample[기본값=1] 0~1
# colsample_bytree [0,0.1,0.2,0.3,0.5,0.7,1]    [기본값=1] 0~1
# colsample_bylevel': [0,0.1,0.2,0.3,0.5,0.7,1] [기본값=1] 0~1
# 'colsample_bynode': [0,0.1,0.2,0.3,0.5,0.7,1] [기본값=1] 0~1
# 'reg_alpha' : [0,0.1 ,0.01, 0.001, 1 ,2 ,10]  [기본값=0] 0~inf /L1 절댓값 가중치 규제 
# 'reg_lambda' : [0,0.1 ,0.01, 0.001, 1 ,2 ,10]  [기본값=1] 0~inf /L2 절댓값 가중치 규제 
# max_delta_step[기본값=0]

parameters = {'n_estimators':[100],
              'learning_rate':[0.3],
              'max_depth': [7],
            #   'gamma' : [1,2,3],
            #   'min_child_weight' : [1],
            #   'subsample' : [1],
            #   'colsample_bytree' : [0.5],
            #   'colsample_bylevel': [1],
            #   'colsample_bynode': [1],
              'alpha' : [1],
              'lambda' : [0]
              } # 디폴트 6 
# 통상 max_depth의 디폴트인 6보다 작을 파라미터를 줄 때 성능이 좋다 -> 너무 깊어지면 훈련 데이터에 특화되어 과적합이 될 수 있다.
# 통상 min_depth의 디폴트인 6보다 큰 파라미터를 줄 때 성능이 좋다


#2. 모델 
xgb = XGBClassifier(random_state=123,
                                   )

model = GridSearchCV(xgb,parameters,cv=kfold,n_jobs=-1)

import time
start_time= time.time()
model.fit(x_train,y_train)
end_time=time.time()-start_time
# model.score(x_test,y_test)
result = model.score(x_test,y_test)

print('최적의 매개변수 : ',model.best_params_)
print('최상의 점수 : ',model.best_score_)
print('model.score : ',result)
print('걸린 시간 : ',end_time)

# 최적의 매개변수 :  {'learning_rate': 0.3, 'n_estimators': 100}
# 최상의 점수 :  0.5517117659153729
# model.score :  0.625
# 걸린 시간 :  7.888291358947754
'''
