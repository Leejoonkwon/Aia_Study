import numpy as np 
import pandas as pd 
from sklearn.datasets import load_breast_cancer,load_iris
from sklearn.datasets import load_diabetes,load_boston
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = load_boston()
x,y = datasets.data, datasets.target # 이렇게도 된다.초등생용 문법
print(x.shape,y.shape) #(506, 13) (506,)

from xgboost import XGBClassifier,XGBRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.ensemble import VotingClassifier,VotingRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
x_train,x_test,y_train,y_test = train_test_split(
    datasets.data,datasets.target,train_size=0.8,shuffle=True,random_state=123,
    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델

xg = XGBRegressor(random_state=123)
lg = LGBMRegressor(random_state=123,
                    learning_rate=0.2)
cat = CatBoostRegressor(verbose=False,random_state=123)
rf =RandomForestRegressor()
models = [('XG', xg), ('LG', lg),('CAT', cat),('RF', rf)]

voting_regressor = VotingRegressor(models, n_jobs=-1)


# hard 는 결과 A 0 B 0 C 1이라면 결과는 0으로 다수결에 따른다.
# soft 는 클래스파이어간의 평균으로 결정

#3. 훈련
voting_regressor.fit(x_train,y_train)

#4. 평가,예측
y_predict = voting_regressor.predict(x_test)

score = r2_score(y_test,y_predict)
print("보팅 결과 :",round(score,4))


Regressor = [cat,xg,lg,rf]
for model2 in Regressor:
    model2.fit(x_train,y_train)
    y_predict = model2.predict(x_test)
    score2 = r2_score(y_test,y_predict)
    class_name = model2.__class__.__name__ # 해당 커맨드로 이름 반환
    print('{0} score : {1:4f}'.format(class_name,score2))
# 보팅 결과 : 0.5273    
# CatBoostRegressor 정확도 : 0.547669
# XGBRegressor 정확도 : 0.461934
# LGBMRegressor 정확도 : 0.469910

######Bagging 후 r2 model xgb
# model.score : 0.8328611676849355

# CatBoostRegressor score : 0.886934
# XGBRegressor score : 0.819620
# LGBMRegressor score : 0.738538
# RandomForestRegressor score : 0.759586



