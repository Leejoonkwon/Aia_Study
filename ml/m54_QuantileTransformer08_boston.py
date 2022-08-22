import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split,KFold 
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import RobustScaler,QuantileTransformer # 이상치에 강함
from sklearn.preprocessing import PowerTransformer
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score,accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from catboost import CatBoostRegressor
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
#1. 데이터
datasets = load_boston()
x, y =datasets.data, datasets.target 
print(x.shape,y.shape) #(506, 13) (506,)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.85,random_state=1234,
)

aaa =[StandardScaler(),MinMaxScaler(),RobustScaler(),
      QuantileTransformer(),PowerTransformer(method='yeo-johnson'),
    #   PowerTransformer(method='box-cox')
      ]
for i in aaa:
    scaler = i
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = RandomForestRegressor()
    model.fit(x_train,y_train)
    y_predict =model.predict(x_test)
    results = r2_score(y_test,y_predict)
    print("{0} : {1:4f} ".format(i,round(results, 4)))

# StandardScaler() : 0.924300 
# MinMaxScaler() : 0.930000 
# RobustScaler() : 0.914500 
# QuantileTransformer() : 0.920100 
# PowerTransformer() : 0.925500 






'''
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
# df['CRIM']= np.log1p(df['CRIM'])      # 로그 변환 결과 : 0.7596
df['ZN']= np.log1p(df['ZN'])          # 로그 변환 결과 : 0.7734
df['TAX']= np.log1p(df['TAX'])        # 로그 변환 결과 : 0.7669
df['B']= np.log1p(df['B'])            # 로그 변환 결과 : 0.7711
# 로그 변환 결과 : 0.7785 (ZN,TAX,B만 하는 것이 좋다)
### np.log1p 공부하기
# print(df['B'].head())

x_train,x_test,y_train,y_test = train_test_split(
    df,y,train_size=0.8,random_state=1234,
)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델
# model = LinearRegression()
model = RandomForestRegressor()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가,예측
y_predict =model.predict(x_test)
results = r2_score(y_test,y_predict)
print("로그 변환 결과 :",round(results, 4))
# LR 기냥 결과 : 0.7665 =>>>> RF 기냥 결과 : 0.9179

# log 후 LR 기냥 결과 : 0.7711
'''
