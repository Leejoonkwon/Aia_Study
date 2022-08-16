from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import r2_score,accuracy_score
import numpy as np
#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
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

# 최적의 매개변수 : RandomForestClassifier(max_depth=6, min_samples_leaf=3, n_jobs=-1)
# 최적의 파라미터 : {'max_depth': 6, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 100, 'n_jobs': -1}  
# best_score : 0.9862068965517242
# model_score : 1.0
# accuracy_score : 1.0
# 최적 튠  ACC : 1.0
# 걸린 시간 : 18.15 초
#============= pipe HalvingGridSearchCV
# 최적의 매개변수 : Pipeline(steps=[('minmax', MinMaxScaler()),
#                 ('RF',
#                  RandomForestClassifier(max_depth=8, min_samples_leaf=3,
#                                         n_jobs=2))])
# 최적의 파라미터 : {'RF__max_depth': 8, 'RF__min_samples_leaf': 3, 'RF__min_samples_split': 2, 
# 'RF__n_estimators': 100, 'RF__n_jobs': 2}      
# best_score : 0.9599130974549969
# model_score : 0.9666666666666667
# accuracy_score : 0.9666666666666667
# 최적 튠  ACC : 0.9666666666666667
# 걸린 시간 : 30.46 초
#============= pipe GridSearchCV
# 최적의 매개변수 : Pipeline(steps=[('minmax', MinMaxScaler()),
#                 ('RF',
#                  RandomForestClassifier(max_depth=8, min_samples_leaf=3,
#                                         min_samples_split=3, n_jobs=-1))])
# 최적의 파라미터 : {'RF__max_depth': 8, 'RF__min_samples_leaf': 3, 'RF__min_samples_split': 3, 
# 'RF__n_estimators': 100, 'RF__n_jobs': -1}     
# best_score : 0.9673175571041422
# model_score : 0.9694444444444444
# accuracy_score : 0.9694444444444444
# 최적 튠  ACC : 0.9694444444444444
# 걸린 시간 : 30.79 초

#============= pipe RandomizedSearchCV
# 최적의 매개변수 : Pipeline(steps=[('minmax', MinMaxScaler()),
#                 ('RF',
#                  RandomForestClassifier(max_depth=8, min_samples_leaf=3,
#                                         min_samples_split=3, n_estimators=200,
#                                         n_jobs=2))])
# 최적의 파라미터 : {'RF__n_jobs': 2, 'RF__n_estimators': 200, 'RF__min_samples_split': 3, 'RF__min_samples_leaf': 3, 'RF__max_depth': 8}      
# best_score : 0.9645228416569882
# model_score : 0.9694444444444444
# accuracy_score : 0.9694444444444444
# 최적 튠  ACC : 0.9694444444444444
# 걸린 시간 : 6.47 초
