import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split,KFold 
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import RobustScaler,QuantileTransformer # 이상치에 강함
from sklearn.preprocessing import PowerTransformer
from sklearn.datasets import load_boston,load_breast_cancer
from sklearn.metrics import r2_score,accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from catboost import CatBoostRegressor
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
#1. 데이터
datasets = load_breast_cancer()
x, y =datasets.data, datasets.target 
print(x.shape,y.shape) #(506, 13) (506,)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8,random_state=1234,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = LogisticRegression()
# model = RandomForestRegressor()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가,예측
y_predict =model.predict(x_test)
results = accuracy_score(y_test,y_predict)
print("기냥 결과 :",round(results, 4))
# 기냥 결과 : 0.7665 =>>>> RF 기냥 결과 : 0.9179
##############로그 변환 ###############
df = pd.DataFrame(datasets.data,columns=[datasets.feature_names])
print(df) # [506 rows x 13 columns]

# df.plot.box()
# plt.title('boston')
# plt.xlabel('Feature')
# plt.ylabel('데이터값')
# plt.show()

# print(df['B'].head())
# log 취하기
# 기냥 결과 : 0.9561
df['mean area']= np.log1p(df['mean area'])     # 로그 변환 결과 : 0.9474
df['worst area']= np.log1p(df['worst area'])       # 로그 변환 결과 : 0.9474
# df['area error']= np.log1p(df['area error'])       # 로그 변환 결과 : 0.9386
# 로그 변환 결과 : 0.9474 
### np.log1p 공부하기
# print(df['B'].head())

x_train,x_test,y_train,y_test = train_test_split(
    df,y,train_size=0.8,random_state=1234,
)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델
model = LogisticRegression()
# model = RandomForestRegressor()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가,예측
y_predict =model.predict(x_test)
results = accuracy_score(y_test,y_predict)
print("로그 변환 결과 :",round(results, 4))
# LR 기냥 결과 : 0.7665 =>>>> RF 기냥 결과 : 0.9179

# log 후 LR 기냥 결과 : 0.7711

