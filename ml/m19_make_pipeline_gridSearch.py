from tabnanny import verbose
import numpy as np 
from sklearn.datasets import load_iris 
from sklearn.preprocessing import MinMaxScaler,StandardScaler 

#1. 데이터 
datasets = load_iris() 
x = datasets.data 
y = datasets.target   
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,KFold,StratifiedKFold

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.8
       ,random_state=1234,shuffle=True)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train) 
# x_test = scaler.transform(x_test)

#2. 모델 
from sklearn.svm import LinearSVC, SVC 
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline,Pipeline 

# model = SVC()
# pipe = make_pipeline(MinMaxScaler(),SVC())
pipe = make_pipeline(StandardScaler(),a = RandomForestClassifier())
# pipe = Pipeline([('minmax',MinMaxScaler()),
#                   ('RF',RandomForestClassifier())
#                   ],verbose=2)
#
# 모델 정의와 스케일링을 정의해주지 않아도  fit에서 fit_transform이 적용된다.
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=100)

parameters = [
    {'a__n_estimators':[100, 200],'a__max_depth':[6, 8],
     'a__min_samples_leaf':[3,5],
     'a__min_samples_split':[2, 3],'a__n_jobs':[-1, 2]},
    {'a__n_estimators':[300, 400],'a__max_depth':[6, 8],
     'a__min_samples_leaf':[7, 10],
     'a__min_samples_split':[4, 7],'a__n_jobs':[-1, 4]}
   
    ]     
# pipe로 했을 때
model = GridSearchCV(pipe,parameters,cv = kfold,refit=True,n_jobs=-1,verbose=1)

#3. 훈련 
model.fit(x_train,y_train)
# fit에서 verbose는 안 먹힌다.
#4. 평가,예측
result = model.score(x_test,y_test)
# 모델 정의와 상관없이 model로 정의된 기능에 x_test에 transform이 적용된다

print('model.score :',result)
