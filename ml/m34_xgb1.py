from unittest import skipUnless
from sklearn.datasets import load_breast_cancer 
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
print(x.shape,y.shape) #(569, 30) (569,)

x_train,x_test,y_train,y_test = train_test_split(x,y,
         shuffle=True,random_state=123,train_size=0.8,stratify=y)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=123)
# parameters = {'n_estimators':[100], # 디폴트 100/ 1~inf 무한대 
# eta[기본값=0.3, 별칭: learning_rate]
# gamma[기본값=0, 별칭: min_split_loss]
# max_depth[기본값=6]
# min_child_weight[기본값=1]
# max_delta_step[기본값=0]
# subsample[기본값=1]

parameters = {'n_estimators':[100,],
              'learning_rate':[0.1 ],
              'max_depth': [3],
              'gamma' : [1],
              'min_child_weight' : [0, 0.1, 0.5, 1, 5, 10, 100]} # 디폴트 6 
# 통상 max_depth의 디폴트인 6보다 작을 파라미터를 줄 때 성능이 좋다 -> 너무 깊어지면 훈련 데이터에 특화되어 과적합이 될 수 있다.
# 통상 min_depth의 디폴트인 6보다 큰 파라미터를 줄 때 성능이 좋다


#2. 모델 
xgb = XGBClassifier(random_state=123,
                    n_estimators=100
                )

model = GridSearchCV(xgb,parameters,cv=kfold,n_jobs=-1)

model.fit(x_train,y_train)

# model.score(x_test,y_test)

print('최적의 매개변수 : ',model.best_params_)
print('최상의 점수 : ',model.best_score_)
###########################################
# 최적의 매개변수 :  {'n_estimators': 100}
# 최상의 점수 :  0.9626373626373625
###########################################
# 최적의 매개변수 :  {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}
# 최상의 점수 :  0.9692307692307693
###########################################
# 최적의 매개변수 :  {'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}
# 최상의 점수 :  0.9736263736263737
###########################################
# 최적의 매개변수 :  {'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 100}
# 최상의 점수 :  0.9736263736263737



