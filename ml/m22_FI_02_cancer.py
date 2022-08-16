import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC 
from sklearn.svm import LinearSVR 
import matplotlib
from tqdm import tqdm
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
#1. 데이터
datasets= load_breast_cancer()
x = datasets['data']
y = datasets['target']
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,KFold,StratifiedKFold

allfeature = round(x.shape[1]*0.2, 0)
print('자를 갯수: ', int(allfeature))


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=1234)

# 2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

models = [DecisionTreeClassifier(), RandomForestClassifier(),
          GradientBoostingClassifier(), XGBClassifier()]


# 3. 컴파일, 훈련, 평가, 예측
for model in models:
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    if str(model).startswith('XGB'):
        print('XGB 의 스코어:        ', score)
    else:
        print(str(model).strip('()'), '의 스코어:        ', score)
        
    featurelist = []
    for a in range(int(allfeature)):
        featurelist.append(np.argsort(model.feature_importances_)[a])
        
    x_bf = np.delete(x, featurelist, axis=1)
    
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_bf, y, shuffle=True, train_size=0.8, random_state=1234)
    model.fit(x_train2, y_train2)
    score = model.score(x_test2, y_test2)
    if str(model).startswith('XGB'):
        print('XGB 의 드랍후 스코어: ', score)
    else:
        print(str(model).strip('()'), '의 드랍후 스코어: ', score)
# 최적의 매개변수 : RandomForestClassifier(max_depth=8, min_samples_leaf=3, n_jobs=2)
# 최적의 파라미터 : {'max_depth': 8, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 100, 'n_jobs': 2}   
# best_score : 0.9582417582417582
# model_score : 0.9649122807017544
# accuracy_score : 0.9649122807017544
# 최적 튠  ACC : 0.9649122807017544

#===============pipeline GridSearchCV
# 최적의 파라미터 : {'RF__max_depth': 8, 'RF__min_samples_leaf': 3, 'RF__min_samples_split': 3, 
# 'RF__n_estimators': 200, 'RF__n_jobs': 2}      
# best_score : 0.9670329670329672
# model_score : 0.9210526315789473
# accuracy_score : 0.9210526315789473
# 최적 튠  ACC : 0.9210526315789473
# 걸린 시간 : 23.27 초
#===============pipeline GridSearchCV
# 최적의 파라미터 : {'RF__n_jobs': -1, 'RF__n_estimators': 300, 'RF__min_samples_split': 7, 'RF__min_samples_leaf': 7, 'RF__max_depth': 8}     
# best_score : 0.9604395604395606
# model_score : 0.9210526315789473
# accuracy_score : 0.9210526315789473
# 최적 튠  ACC : 0.9210526315789473
# 걸린 시간 : 5.8 초 

#===============pipeline HalvingGridSearchCV
# 최적의 매개변수 : Pipeline(steps=[('minmax', MinMaxScaler()),
#                 ('RF',
#                  RandomForestClassifier(max_depth=8, min_samples_leaf=3,
#                                         n_jobs=2))])
# 최적의 파라미터 : {'RF__max_depth': 8, 'RF__min_samples_leaf': 3, 'RF__min_samples_split': 2, 
# 'RF__n_estimators': 100, 'RF__n_jobs': 2}      
# best_score : 0.9444444444444444
# model_score : 0.9210526315789473
# accuracy_score : 0.9210526315789473
# 최적 튠  ACC : 0.9210526315789473
# 걸린 시간 : 26.3 초
