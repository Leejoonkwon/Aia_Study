import numpy as np  
from sklearn.datasets import load_diabetes    


#1. 데이터
datasets = load_diabetes()
x =  datasets.data 
y =  datasets.target   

from sklearn.model_selection import train_test_split 

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.8,shuffle=True,random_state=123)


#2. 모델 구성
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor 
from xgboost import XGBClassifier,XGBRegressor 

model = DecisionTreeRegressor() 
# model = RandomForestRegressor() 
# model  = GradientBoostingRegressor()
# model  = XGBRegressor()
#3. 훈련 
model.fit(x_train,y_train)

#4. 평가,예측

result = model.score(x_test,y_test)
print("model.score :",result)

from sklearn.metrics import accuracy_score, r2_score 
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print('r2_score :',r2)


print("=============================")
print(model,':',model.feature_importances_)
import matplotlib.pyplot as plt

# def plot_feature_importances(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features),model.feature_importances_,align='center')
#     plt.yticks(np.arange(n_features),datasets.feature_names)
#     plt.xlabel('Feature_importances')
#     plt.ylabel('Features')
#     plt.ylim(-1,n_features)
#     plt.title(model)
def plot_feature_importances(model):
    n_features = datasets.data.shape[1]
    plt.pie(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),datasets.feature_names)
    plt.xlabel('Feature_importances')
    plt.ylabel('Features')
    plt.ylim(-1,n_features)
    plt.title(model)
ratio = model.feature_importances_
labels = ['Apple', 'Banana', 'Melon', 'Grapes']

plt.pie(ratio, labels=labels, autopct='%.1f%%')
plt.show()

plot_feature_importances(model)
plt.show()


# feature_importtances는 y값에 영향을 미치는 정도를 수치화한다.
# iris는 컬럼이 4개지만 데이터가 더 방대해질경우 필요한 것과 불필요한 것을 분리할 수 있다.
# 여러가지 모델을 비교해서 영향을 확인해야한다.
# DecisionTreeRegressor() : [0.10435529 0.02171698 0.22898906 0.05458676 0.03858524 
#                            0.0525853  0.03863638 0.02242778 0.36608123 0.07203599]
# model.score : 0.16460218742421662
# r2_score : 0.16460218742421662

# RandomForestRegressor() : [0.05645885 0.00990777 0.27930196 0.10773298 0.0406001  
#                            0.05511041 0.05479674 0.02652711 0.27875095 0.09081314] 
# model.score : 0.5285162928080618
# r2_score : 0.5285162928080618

# GradientBoostingRegressor() : [0.04983822 0.01079916 0.30199132 0.11189149 0.02805566 
#                                0.0581757  0.04065599 0.01636338 0.33859948 0.04362959]
# model.score : 0.5575872699933667
# r2_score : 0.5575872699933667


# XGBRegressor() : [0.03234756 0.0447546  0.21775807 0.08212128 0.04737141 
#                   0.04843819 0.06012432 0.09595273 0.30483875 0.06629313]
# model.score : 0.4590400803596264
# r2_score : 0.4590400803596264



