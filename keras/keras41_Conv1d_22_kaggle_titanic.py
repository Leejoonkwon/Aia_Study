# [실습]
# 시작!!!
# datasets.descibe()
# datasets.info()
# datasets.isnull().sum()

# pandas의  y 라벨의 종류가 무엇인지 확인하는 함수 쓸것
# numpy 에서는 np.unique(y,return_counts=True)

import numpy as np
import pandas as pd 
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten,LSTM,Conv1D
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MaxAbsScaler,RobustScaler #직접 찾아라!

from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import seaborn as sns

#1.데이터

path = './_data/kaggle_titanic/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)
print(train_set) # [891 rows x 11 columns]
print(train_set.describe())
print(train_set.info())

# f, ax = plt.subplots(1,2,figsize=(18,8))
# train_set['Survived'].value_counts().plot.pie(explode = [0,0.1], autopct = '%1.1f%%', ax=ax[0], shadow = True)
# ax[0].set_title('Survived')
# ax[0].set_ylabel('')
# sns.countplot('Survived', data = train_set, ax = ax[1])
# ax[1].set_title('Survived')
# plt.show()
# f, ax = plt.subplots(1,2,figsize=(18,8))
# train_set[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
# ax[0].set_title('Survived vs Sex')
# sns.countplot('Sex', hue = 'Survived', data = train_set, ax = ax[1])
# ax[1].set_title('Sex:Survived vs Dead')
# plt.show()

print(test_set) # [418 rows x 10 columns]
print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
# Survived      0
# Pclass        0
# Name          0
# Sex           0
# Age         177
# SibSp         0
# Parch         0
# Ticket        0
# Fare          0
# Cabin       687
# Embarked      2

# train_set = train_set.fillna(train_set.median())
print(test_set.isnull().sum())
# Pclass        0
# Name          0
# Sex           0
# Age          86
# SibSp         0
# Parch         0
# Ticket        0
# Fare          1
# Cabin       327
# Embarked      0

drop_cols = ['Cabin']
train_set.drop(drop_cols, axis = 1, inplace =True)
test_set = test_set.fillna(test_set.mean())
train_set['Embarked'].fillna('S')
train_set = train_set.fillna(train_set.mean())

print(train_set) 
print(train_set.isnull().sum())

test_set.drop(drop_cols, axis = 1, inplace =True)
cols = ['Name','Sex','Ticket','Embarked']
for col in cols:
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])
x = train_set.drop(['Survived'],axis=1) #axis는 컬럼 
print(x) #(891, 9)
y = train_set['Survived']
print(y.shape) #(891,)


# test_set.drop(drop_cols, axis = 1, inplace =True)
gender_submission = pd.read_csv(path + 'gender_submission.csv',#예측에서 쓸거야!!
                       index_col=0)
# y의 라벨값 : (array([0, 1], dtype=int64), array([549, 342], dtype=int64))

###########(pandas 버전 원핫인코딩)###############
# y_class = pd.get_dummies((y))
# print(y_class.shape) # (891, 2)

# 해당 기능을 통해 y값을 클래스 수에 맞는 열로 늘리는 원핫 인코딩 처리를 한다.
#1개의 컬럼으로 [0,1,2] 였던 값을 ([1,0,0],[0,1,0],[0,0,1]과 같은 shape로 만들어줌)



x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8,shuffle=True ,random_state=100)
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#셔플을 False 할 경우 순차적으로 스플릿하다보니 훈련에서는 나오지 않는 값이 생겨 정확도가 떨어진다.
#디폴트 값인  shuffle=True 를 통해 정확도를 올린다.
print(x_train.shape) #(712, 9)
print(x_test.shape) #(179, 9)

x_train = x_train.reshape(712, 9,1)
x_test = x_test.reshape(179, 9,1)
y_train = pd.get_dummies((y_train)) 
y_test = pd.get_dummies((y_test))
print(y_test.shape)

#2. 모델 구성

model = Sequential()
# model.add(LSTM(10,input_shape=(9,1)))
model.add(Conv1D(10,2,input_shape=(9,1)))
model.add(Flatten())
model.add(Dense(16,activation='swish'))
# model.add(Dropout(0.3))
model.add(Dense(16,activation='swish'))
# model.add(Dropout(0.3))
model.add(Dense(16,activation='swish'))
# model.add(Dropout(0.3))
model.add(Dense(16,activation='swish'))
# model.add(Dropout(0.3))
model.add(Dense(16,activation='swish'))
# model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(2, activation='sigmoid'))
#다중 분류로 나오는 아웃풋 노드의 개수는 y 값의 클래스의 수와 같다.활성화함수 'softmax'를 통해 
# 아웃풋의 합은 1이 된다.
import datetime
date = datetime.datetime.now()
print(date)

date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)
import time
start_time = time.time()


# #3. 컴파일,훈련
filepath = './_ModelCheckPoint/K24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
#04d :                  4f : 
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
earlyStopping = EarlyStopping(monitor='loss', patience=100, mode='min', 
                              verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
                      save_best_only=True, 
                      filepath="".join([filepath,'k25_', date, '_kaggle_titanic_', filename])
                    )
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=30, 
                validation_split=0.3,
                callbacks = [earlyStopping],
                verbose=2
                )
# 다중 분류 모델은 'categorical_crossentropy'만 사용한다 !!!!

#4.  평가,예측

loss,acc = model.evaluate(x_test,y_test)
print('loss :',loss)

y_predict = model.predict(x_test)
print(y_predict) 
y_predict[(y_predict<0.5)] = 0  
y_predict[(y_predict>=0.5)] = 1  
print(y_predict) 
print(y_test.shape) #(134,)

acc = accuracy_score(y_test, y_predict)
print('acc 스코어 :', acc)
end_time =time.time()-start_time
print("걸린 시간 :",end_time)

# y_summit = model.predict(test_set)


# gender_submission['Survived'] = y_summit
# submission = gender_submission.fillna(gender_submission.mean())

# submission [(submission <0.5)] =0
# submission [(submission >=0.5)] =1 
# submission.astype(int)
# submission.to_csv('test21.csv',index=True)

# drop 아웃 전 
# loss : 2.1319031715393066
# acc 스코어 : 0.7821229050279329
# drop 아웃 후
# loss : 0.593296468257904
# acc 스코어 : 0.8268156424581006

#cnn dnn 후
# loss : 0.5045608282089233
# racc 스코어 : 0.7877094972067039
#######LSTM
# loss : 0.4824408292770386
# acc 스코어 : 0.7094972067039106
#######Conv1d
# loss : 0.4539312422275543
# acc 스코어 : 0.8044692737430168
# 걸린 시간 : 16.09059190750122
