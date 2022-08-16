# m20_04

import numpy as np
from sklearn.datasets import load_diabetes
import pandas as pd

#1. 데이터
datasets = load_diabetes()
print(datasets['feature_names'])
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
x = datasets.data
y = datasets.target

x = pd.DataFrame(x, columns=[['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=123)

#2. 모델구성
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor,XGBClassifier

# model = DecisionTreeRegressor()
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
model = XGBRegressor()


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print("model.score : ", result)

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2_score : ', r2)

print("=====================================")
print(model, ':', model.feature_importances_)

import matplotlib.pyplot as plt
# def plot_feature_importances(model) :
#     n_features = datasets.data.shape[1] # dataset 데이터의 열의 개수
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel('Feature Importances')
#     plt.ylabel('Features')
#     plt.ylim(-1, n_features)
#     plt.title(model)
    
# plot_feature_importances(model)
# plt.show()

from xgboost.plotting import plot_importance
plot_importance(model)
plt.show()

# 현재 넘파이로 받아서 y축에 f0...으로 뜨지만 판다스로 받으면 feature name이 뜬다.
# x축 tree 사용 빈도수, 다 더한 상태에서 나누기 N









# DecisionTreeRegressor() : 
# model.score :  0.1281522909778141
# r2_score :  0.1281522909778141
# [0.09784516 0.02173979 0.23345315 0.05219251 0.03824603 0.0579618
#  0.04904269 0.01366745 0.3619148  0.07393661]

# RandomForestRegressor() : 
# model.score :  0.528126517540028     
# r2_score :  0.528126517540028     
# [0.05729019 0.01028294 0.30025778 0.10169514 0.04171223 0.05340497
#  0.05372396 0.0257577  0.27685019 0.0790249 ]

# GradientBoostingRegressor() : 
# model.score :  0.5546481119433642
# r2_score :  0.5546481119433642
# [0.04958545 0.01079504 0.30351231 0.11174848 0.02822866 0.05416627
#  0.03985299 0.01896411 0.33862935 0.04451734]

# XGBRegressor() :
# model.score :  0.4590400803596264
# r2_score :  0.4590400803596264
# [0.03234756 0.0447546  0.21775807 0.08212128 0.04737141 0.04843819
#  0.06012432 0.09595273 0.30483875 0.06629313]



# 피처의 중요도에 따라 전체 피처를 다 쓸 필요가 있는지에 대한 고민
# 통상적으로 XGBOOST 가 제일 좋은 성능을 보여주긴 하지만, feature importances를 통해 확인은 필요