# 실습
# 피처임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거하여
# 데어터셋 재구성후
# 각 모델별로 돌려서 결과 도출!
import numpy as np  
from sklearn.datasets import load_iris    


#1. 데이터
datasets = load_iris()

x =  datasets.data 
y =  datasets.target   
# print(datasets['feature_names'])
aaa = np.round(len(datasets['feature_names'])*0.25,0)
print(aaa)

from sklearn.model_selection import train_test_split 

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.8,shuffle=True,random_state=123)

#2. 모델 구성
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier 
from xgboost import XGBClassifier

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


#empty list for progress bar in tqdm library
for model in (model_list):
    empty_list.append(model)
    print(empty_list)

    clf = models(model)    
    #classifier
    clf.fit(x_train, y_train) 
    result = clf.score(x_test,y_test)
    abc = np.argsort(clf.feature_importances_,axis=0)[aaa] # [0,7,2,3,5]
    print('{}-{}'.format(model,result))
    for i in aaa:
        x = np.delete(x,i,axis=1)
        x1_train,x1_test,y1_train,y1_test = train_test_split(x,y,train_size=0.75,shuffle=True,random_state=100)
        model_list3 = ['RF', 'GB',  'XGB', 'DT']
        empty_list2 = []
        for model2 in (model_list3):
            empty_list2.append(model2)
            clf2 = models(model2)    
            #classifier
            clf2.fit(x1_train, y1_train) 
            #Predict
            result2 = clf2.score(x1_test,y1_test)
            pred = clf2.predict(x1_test) 
            print('{}-{}'.format(model2,result2))


