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

from sklearn.model_selection import train_test_split 

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.8,shuffle=True,random_state=123)


#2. 모델 구성
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier 
from xgboost import XGBClassifier

def models(model):
    if model == 'rf':
        mod = RandomForestClassifier()
    elif model == 'gb':
        mod = GradientBoostingClassifier()
    elif model == 'xgb':
        mod =  XGBClassifier()
    elif model == 'dt':
        mod =  DecisionTreeClassifier()
  
    return mod
model_list = ['rf', 'gb',  'xgb', 'dt']
empty_list = [] #empty list for progress bar in tqdm library
for model in (model_list):
    empty_list.append(model) # fill empty_list to fill progress bar
    #classifier
    clf = models(model)
    #Training
    clf.fit(x_train, y_train) 
    #Predict
    result = clf.score(x_test,y_test)
    pred = clf.predict(x_test) 
    print('{}-{}'.format(model,result))
    print(model,':',np.argsort(clf.feature_importances_,axis=0),len(clf.feature_importances_)*0.25)
    
#4. 평가,예측

'''
# DecisionTreeClassifier() : [0.01088866 0.01253395 0.55770372 0.41887367]
# RandomForestClassifier() : [0.09711772 0.02531929 0.44701223 0.43055075]
# GradientBoostingClassifier() : [0.00085471 0.02276816 0.56079501 0.41558212]
# XGBClassifier() : [0.0089478  0.01652037 0.75273126 0.22180054]

import matplotlib.pyplot as plt
def plot_feature_importances(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),datasets.feature_names)
    plt.xlabel('Feature_importances')
    plt.ylabel('Features')
    plt.ylim(-1,n_features)
   

    
plt.subplot(2,2,1)
plt.title('DecisionTreeClassifier')
plot_feature_importances(model1)

plt.subplot(2,2,2)
plt.title('RandomForestClassifier')
plot_feature_importances(model2)

plt.subplot(2,2,3)
plt.title('GradientBoostingClassifier')
plot_feature_importances(model3)

plt.subplot(2,2,4)
plt.title('XGBClassifier')
plot_feature_importances(model4)

plt.show()

#4. 평가,예측
abc = [model1,model2,model3,model4]
for i in abc:
    result = abc[i].score(x_test,y_test)
    print("model.score :",result)
'''
