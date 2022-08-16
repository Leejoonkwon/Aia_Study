from os import name
from tabnanny import verbose
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
from xgboost import XGBClassifier,XGBClassifier 

model1 = DecisionTreeClassifier() 
model2 = RandomForestClassifier() 
model3  = GradientBoostingClassifier()
model4  = XGBClassifier()
# #3. 훈련 
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train,verbose=2)

#4. 평가,예측

# result = model.score(x_test,y_test)
# print("model.score :",result)

# from sklearn.metrics import accuracy_score, r2_score 
# y_predict = model.predict(x_test)
# r2 = r2_score(y_test,y_predict)
# print('r2_score :',r2)


# print("=============================")
print(model1,':',model1.feature_importances_)
print(model2,':',model2.feature_importances_)
print(model3,':',model3.feature_importances_)
print(model4,':',model4.feature_importances_)
import matplotlib.pyplot as plt
def plot_feature_importances(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),datasets.feature_names)
    plt.xlabel('Feature_importances')
    plt.ylabel('Features')
    plt.ylim(-1,n_features)
    plt.title(model)

      


plt.subplot(2,2,1)
plot_feature_importances(model2)

plt.subplot(2,2,2)
plot_feature_importances(model2)

plt.subplot(2,2,3)
plot_feature_importances(model3)

plt.subplot(2,2,4)
plot_feature_importances(model4)

plt.show()
# ax1.plot(x)
# ax2.bar(x)
# plt.show()
# plt.subplot('맹그러봐')
# plt.subplot([plot_feature_importances(model1),
#              plot_feature_importances(model2),
#              plot_feature_importances(model3)])
# plt.subplot(plot_feature_importances([model1,model2,model3]))


# plot_feature_importances(model1)

# plt.show()


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



