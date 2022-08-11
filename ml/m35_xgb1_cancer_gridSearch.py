from unittest import skipUnless
from sklearn.datasets import load_breast_cancer,load_diabetes
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler 
from xgboost import XGBClassifier,XGBRegressor
import time 

#1.데이터 
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape,y.shape) #(442, 10) (442,)

x_train,x_test,y_train,y_test = train_test_split(x,y,
         shuffle=True,random_state=123,train_size=0.8)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits,shuffle=True,random_state=123)
# parameters = {'n_estimators':[100,200,300,400,500,1000], # 디폴트 100/ 1~inf 무한대 
# eta[기본값=0.3, 별칭: learning_rate] learning_rate':[0.1,0.2,0.3,0.4,0.5,0.7,1]
# max_depth': [1,2,3,4,5,6,7][기본값=6]
# gamma[기본값=0, 별칭: min_split_loss] [0,0.1,0.3,0.5,0.7,0.8,0.9,1]
# min_child_weight[기본값=1] 0~inf
# subsample[기본값=1][0,0.1,0.3,0.5,0.7,1] 0~1
# colsample_bytree [0,0.1,0.2,0.3,0.5,0.7,1]    [기본값=1] 0~1
# colsample_bylevel': [0,0.1,0.2,0.3,0.5,0.7,1] [기본값=1] 0~1
# 'colsample_bynode': [0,0.1,0.2,0.3,0.5,0.7,1] [기본값=1] 0~1
# 'reg_alpha' : [0,0.1 ,0.01, 0.001, 1 ,2 ,10]  [기본값=0] 0~inf /L1 절댓값 가중치 규제 
# 'reg_lambda' : [0,0.1 ,0.01, 0.001, 1 ,2 ,10]  [기본값=1] 0~inf /L2 절댓값 가중치 규제 
# max_delta_step[기본값=0]

# parameters = {'n_estimators':[100,200], # 2
#               'learning_rate':[0.1,0.3], # 3
#               'max_depth': [2,3,4], # 3
#               'gamma' : [0.1,1], # 4 
#               'min_child_weight' : [5], # 1
#               'subsample' : [0.5,0.7,1], # 4
#               'colsample_bytree' : [0.3,0.5,1], # 4
#               'colsample_bylevel': [0.3,0.5,1],# 4
#               'colsample_bynode': [0.3,0.5,1],# 4
#               'alpha' : [ 1 ,2 ,10], # 3
#               'lambda' : [1 ,2 ,10] # 3
#               } # 디폴트 6 
parameters ={'alpha': [1], 'colsample_bylevel': [0.3], 'colsample_bynode': [1], 
 'colsample_bytree': [0.5], 'gamma': [1], 'lambda': [2], 'learning_rate': [0.3], 
 'max_depth': [2], 'min_child_weight': [5], 'n_estimators': [100], 'subsample': [0.7]}
# 통상 max_depth의 디폴트인 6보다 작을 파라미터를 줄 때 성능이 좋다 -> 너무 깊어지면 훈련 데이터에 특화되어 과적합이 될 수 있다.
# 통상 min_depth의 디폴트인 6보다 큰 파라미터를 줄 때 성능이 좋다


#2. 모델 
xgb = XGBClassifier(random_state=123,
                    n_estimators=100,
                    # tree_method='gpu_hist'
                )

model = GridSearchCV(xgb,parameters,cv=kfold,n_jobs=-1)
import time
start_time = time.time()
model.fit(x_train,y_train)
end_time =time.time()-start_time
# model.score(x_test,y_test)
results = model.score(x_test,y_test)
print('최적의 매개변수 : ',model.best_params_)
print('최상의 점수 : ',model.best_score_)
print('model.socre : ',results)
print('걸린 시간 : ',end_time)

####################################################
# 최적의 매개변수 :  {'learning_rate': 0.2, 'n_estimators': 100}
# 최상의 점수 :  0.25834227216709194
# model.socre :  0.46723494962000645
####################################################
# 최적의 매개변수 :  {'learning_rate': 0.2, 'max_depth': 1, 'n_estimators': 100}
# 최상의 점수 :  0.40412545445713616
# model.socre :  0.5994781962366296
####################################################
# 최적의 매개변수 :  {'gamma': 0, 'learning_rate': 0.2, 'max_depth': 1, 'n_estimators': 100}
# 최상의 점수 :  0.40412545445713616
# model.socre :  0.5994781962366296
####################################################
# 최적의 매개변수 :  {'gamma': 0, 'learning_rate': 0.2, 'max_depth': 1, 'min_child_weight': 5, 'n_estimators': 100}
# 최상의 점수 :  0.41199505440070894
# model.socre :  0.597604046868436
####################################################
# 최적의 매개변수 :  {'gamma': 0, 'learning_rate': 0.2, 'max_depth': 1, 'min_child_weight': 5, 'n_estimators': 100, 'subsample': 1}
# 최상의 점수 :  0.41199505440070894
# model.socre :  0.597604046868436
####################################################
# 최적의 매개변수 :  {'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.2, 
# 'max_depth': 1, 'min_child_weight': 5, 'n_estimators': 100, 'subsample': 1}
# 최상의 점수 :  0.41199505440070894
# model.socre :  0.597604046868436
# 최적의 매개변수 :  {'alpha': 1, 'colsample_bylevel': 0.3, 
# 'colsample_bynode': 1, 'colsample_bytree': 0.5, 'gamma': 1, 'lambda': 2, 
# 'learning_rate': 0.3, 'max_depth': 2, 'min_child_weight': 5, 'n_estimators': 100, 'subsample': 0.7}
# 최상의 점수 :  0.9670329670329672
# model.socre :  0.9736842105263158
# 걸린 시간 :  1.7486727237701416