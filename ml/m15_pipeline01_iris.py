import numpy as np 
from sklearn.datasets import load_iris 
from sklearn.preprocessing import MinMaxScaler,StandardScaler 

#1. 데이터 
datasets = load_iris() 
x = datasets.data 
y = datasets.target   

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.8
       ,random_state=1234,shuffle=True)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train) 
# x_test = scaler.transform(x_test)

#2. 모델 
from sklearn.svm import LinearSVC, SVC 
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline 

# model = SVC()
# model = make_pipeline(MinMaxScaler(),SVC())
model = make_pipeline(MinMaxScaler(),RandomForestClassifier())
# 모델 정의와 스케일링을 정의해주지 않아도  fit에서 fit_transform이 적용된다.
from sklearn.utils import all_estimators

allAlgorithms = all_estimators(type_filter='regressor')
# allAlgorithms = all_estimators(type_filter='Regressor')

# print('allAlgorithms :',allAlgorithms)
print('모델의 갯수 :',len(allAlgorithms)) #모델의 갯수 : 41

for (name,algorithms) in allAlgorithms:
    try: # for문을 실행하는 와중에 예외 (error)가 발생하면 무시하고 진행 <예외처리>
        model = make_pipeline(MinMaxScaler(),algorithms())

        model.fit(x_train,y_train)
        result = model.score(x_test,y_test)
        print('{}-{}'.format(name,result))
       
    except:
        continue
#3. 훈련 
# model.fit(x_train,y_train)

# #4. 평가,예측
# result = model.score(x_test,y_test)
# 모델 정의와 상관없이 model로 정의된 기능에 x_test에 transform이 적용된다

# print('model.score :',result)
