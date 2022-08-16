#  과제
# activation : sigmoid,relu,linear
# metrics 추가
# EarlyStopping  넣고
# 성능비교
# 감상문 2줄이상!
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten,LSTM,Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
import pandas as pd
from sklearn.svm import LinearSVC 
from sklearn.svm import LinearSVR 
import numpy as np 
#1. 데이터
path = 'D:\study_data\_data\_csv\_ddarung/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
print(train_set)

print(train_set.shape) #(1459, 10)

test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)
submission = pd.read_csv(path + 'submission.csv',#예측에서 쓸거야!!
                       index_col=0)
                       
print(test_set)
print(test_set.shape) #(715, 9) #train_set과 열 값이 '1'차이 나는 건 count를 제외했기 때문이다.예측 단계에서 값을 대입

print(train_set.columns)
print(train_set.info()) #null은 누락된 값이라고 하고 "결측치"라고도 한다.
print(train_set.describe()) 

###### 결측치 처리 1.제거##### dropna 사용
print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
train_set = train_set.fillna(train_set.median())
print(train_set.isnull().sum())
print(train_set.shape)
test_set = test_set.fillna(test_set.median())

x = train_set.drop(['count'],axis=1) #axis는 컬럼 
print(x.columns)
print(x.shape) #(1459, 9)

y = train_set['count']
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,KFold,StratifiedKFold
x = np.array(x)
y = np.array(y)
allfeature = round(x.shape[1]*0.2, 0)
print('자를 갯수: ', int(allfeature))


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.7, random_state=1234)

# 2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# models = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]
def models(model):
    if model == 'RF':
        mod = RandomForestClassifier()
    elif model == 'GB':
        mod = GradientBoostingClassifier()
    elif model == 'XGB':
        mod =  XGBClassifier()
    elif model == 'DT':
        mod =  DecisionTreeClassifier()
  
    return mod
model_list = ['RF', 'GB',  'XGB', 'DT']
empty_list = []

for model in models:
    empty_list.append(model)
    clf = models(model)    
    clf.fit(x_train, y_train) 
    result = clf.score(x_test,y_test) 
    if str(model).startswith('XGB'):
        print('XGB 의 스코어:        ', score)
    else:
        print(str(model).strip('()'), '의 스코어:        ', score)
        
    featurelist = []
    for a in range(int(allfeature)):
        featurelist.append(np.argsort(model.feature_importances_)[a])
        
    x_bf = np.delete(x, featurelist, axis=1)
    
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_bf, y, shuffle=True, train_size=0.7, random_state=1234)
    model.fit(x_train2, y_train2)
    score = model.score(x_test2, y_test2)
    if str(model).startswith('XGB'):
        print('XGB 의 드랍후 스코어: ', score)
    else:
        print(str(model).strip('()'), '의 드랍후 스코어: ', score)
        
#============= pipe HalvingGridSearchCV
# 최적의 매개변수 : Pipeline(steps=[('minmax', MinMaxScaler()),
#                 ('RF',
#                  RandomForestRegressor(max_depth=6, min_samples_leaf=5,
#                                        min_samples_split=3, n_jobs=2))])
# 최적의 파라미터 : {'RF__max_depth': 6, 'RF__min_samples_leaf': 5, 'RF__min_samples_split': 3, 
# 'RF__n_estimators': 100, 'RF__n_jobs': 2}      
# best_score : 0.7507959223647331
# model_score : 0.7591141162607191
# accuracy_score : 0.7591141162607191
# 최적 튠  ACC : 0.7591141162607191
# 걸린 시간 : 25.3 초
#============= pipe GridSearchCV
# 최적의 매개변수 : Pipeline(steps=[('minmax', MinMaxScaler()),
#                 ('RF',
#                  RandomForestRegressor(max_depth=8, min_samples_leaf=3,
#                                        min_samples_split=3, n_estimators=200,
#                                        n_jobs=-1))])
# 최적의 파라미터 : {'RF__max_depth': 8, 'RF__min_samples_leaf': 3, 'RF__min_samples_split': 3, 
# 'RF__n_estimators': 200, 'RF__n_jobs': -1}     
# best_score : 0.7655431824311288
# model_score : 0.7796496700810223
# accuracy_score : 0.7796496700810222
# 최적 튠  ACC : 0.7796496700810222
# 걸린 시간 : 30.97 초

#============= pipe RandomizedSearchCV
# 최적의 매개변수 : Pipeline(steps=[('minmax', MinMaxScaler()),
#                 ('RF',
#                  RandomForestRegressor(max_depth=8, min_samples_leaf=3,
#                                        min_samples_split=3, n_estimators=200,
#                                        n_jobs=-1))])
# 최적의 파라미터 : {'RF__n_jobs': -1, 'RF__n_estimators': 200, 'RF__min_samples_split': 3, 'RF__min_samples_leaf': 3, 'RF__max_depth': 8}     
# best_score : 0.7655627301006306
# model_score : 0.7795832657214612
# accuracy_score : 0.7795832657214612
# 최적 튠  ACC : 0.7795832657214612
# 걸린 시간 : 6.5 초
    
    