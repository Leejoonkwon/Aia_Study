#  과제
# activation : sigmoid,relu,linear
# metrics 추가
# EarlyStopping  넣고
# 성능비교
# 감상문 2줄이상!
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import null
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm_notebook
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
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
print(train_set.shape) #(1460,76)
print(test_set.shape) #(1459, 75) #train_set과 열 값이 '1'차이 나는 건 count를 제외했기 때문이다.예측 단계에서 값을 대입

print(train_set.columns)
print(train_set.info()) #null은 누락된 값이라고 하고 "결측치"라고도 한다.
print(train_set.describe()) 

###### 결측치 처리 1.제거##### dropna 사용
print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
train_set = train_set.fillna(train_set.median())
print(train_set.isnull().sum())
print(train_set.shape)
test_set = test_set.fillna(test_set.median())

x = train_set.drop(['SalePrice'],axis=1) #axis는 컬럼 
print(x.columns)
print(x.shape) #(1460, 75)

y = train_set['SalePrice']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.89, shuffle = True, random_state = 68
 )
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(y)
print(y.shape) # (1460,)


#2. 모델구성
# model = Sequential()
# model.add(Dense(100,input_dim=75))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(1))
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input
input1 = Input(shape=(75,))
dense1 = Dense(100)(input1)
dense2 = Dense(100,activation='relu')(dense1)
dense3 = Dense(100,activation='relu')(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1)
import time
start_time = time.time()
#3. 컴파일,훈련
earlyStopping = EarlyStopping(monitor='loss', patience=100, mode='min', 
                              verbose=1,restore_best_weights=True)
model.compile(loss='mae', optimizer='adam')

hist = model.fit(x_train, y_train, epochs=1010, batch_size=100, 
                validation_split=0.3,
                callbacks = [earlyStopping],
                verbose=2
                )


#verbose = 0으로 할 시 출력해야할 데이터가 없어 속도가 빨라진다.강제 지연 발생을 막는다.



#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)
end_time = time.time()-start_time
print("걸린시간 :",end_time)
# print("============")
# print(hist) #<tensorflow.python.keras.callbacks.History object at 0x000001B6B522F0A0>
# print("============")
# # print(hist.history) #(지정된 변수,history)를 통해 딕셔너리 형태의 데이터 확인 가능 
# print("============")
# print(hist.history['loss'])
# print("============")
# print(hist.history['val_loss'])
# # y_predict = model.predict(x_test)
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'],marker='.',c='red',label='loss') #순차적으로 출력이므로  y값 지정 필요 x
# plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss')
# plt.grid()
# plt.title('영어싫어') #맥플러립 한글 깨짐 현상 알아서 해결해라 
# plt.ylabel('loss')
# plt.xlabel('epochs')
# # plt.legend(loc='upper right')
# plt.legend()
# plt.show()

# loss : 356.1080
# RMSE :  50.77395531000674
###################
# activation 및 EarlyStopping 적용 
# loss : 97.37265014648438
# r2스코어 : 0.36341653990677303

#####################################
# EarlyStopping  적용 및 활성화 함수
# loss : 30.648914337158203
# r2스코어 : 0.7276545891537277
##################
#1. 스케일러 하기전
# loss : 30.648914337158203
# r2스코어 : 0.7276545891537277
##################
#2. MinMaxScaler
# loss : 18206.8515625
# r2스코어 : 0.8657327208251866
# 걸린시간 : 55.83432698249817
##################
#3. StandardScaler
# loss : 16617.701171875
# r2스코어 : 0.8774402652077372
# 걸린시간 : 58.85835576057434
##################
#4. MaxAbsScaler
# loss : 18918.541015625
# r2스코어 : 0.8604294545237556
# 걸린시간 : 58.26255655288696
##################
#5. RobustScaler
# loss : 18403.46875
# r2스코어 : 0.8710011500655894
# 걸린시간 : 41.32139444351196
##################
#6. Sequential에서  함수형 모델로
# loss : 20911.396484375
# r2스코어 : 0.8532889404841759
# 걸린시간 : 26.55862069129944
