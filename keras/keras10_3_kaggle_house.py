# 데이콘 따릉이 문제풀이
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import null
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm_notebook

#1. 데이터
path = './_data/kaggle_house/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)
drop_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
test_set.drop(drop_cols, axis = 1, inplace =True)
submission = pd.read_csv(path + 'submission.csv',#예측에서 쓸거야!!
                       index_col=0)
print(train_set)

print(train_set.shape) #(1459, 10)

train_set.drop(drop_cols, axis = 1, inplace =True)
cols = ['MSZoning', 'Street','LandContour','Neighborhood','Condition1','Condition2',
                'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation',
                'Heating','GarageType','SaleType','SaleCondition','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                'BsmtFinType2','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',
                'FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive','LotShape',
                'Utilities','LandSlope','BldgType','HouseStyle','LotConfig']

for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])


print(test_set)
print(test_set.shape) #(715, 9) #train_set과 열 값이 '1'차이 나는 건 count를 제외했기 때문이다.예측 단계에서 값을 대입

print(train_set.columns)
print(train_set.info()) #null은 누락된 값이라고 하고 "결측치"라고도 한다.
print(train_set.describe()) 

###### 결측치 처리 1.제거##### dropna 사용
print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
train_set = train_set.fillna(train_set.mean())
print(train_set.isnull().sum())
print(train_set.shape)
test_set = test_set.fillna(test_set.mean())

x = train_set.drop(['SalePrice'],axis=1) #axis는 컬럼 
print(x.columns)
print(x.shape) #(1460, 75)

y = train_set['SalePrice']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.84, shuffle = True, random_state = 100
 )
print(y)
print(y.shape) # (1460,)


#2. 모델구성

model = Sequential()
model.add(Dense(100,input_dim=75))
model.add(Dense(100, activation='swish'))
model.add(Dense(100, activation='swish'))
model.add(Dense(100, activation='swish'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
history = model.fit(x_train, y_train , epochs =5450,validation_split=0.25, batch_size=130, verbose=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :',loss)

y_predict = model.predict(x_test) #훈련으로 예측된 값 y_predict와 원래 테스트 값 y_test와 비교

def RMSE(y_test, y_predict):
     return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE :",rmse)  
y_summit = model.predict(test_set)
# print(y_summit)
# print(y_summit.shape)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(['train_set', 'test_set'], loc='upper left')
plt.show()
# loss : 18704.109375
# RMSE : 25966.91846774703
# train_size = 0.949, shuffle = True, random_state = 100
# epochs =2350,validation_split=0.3, batch_size=150, verbose=2


submission['SalePrice'] = y_summit
submission = submission.fillna(submission.mean())
submission.to_csv('test18.csv',index=True)


