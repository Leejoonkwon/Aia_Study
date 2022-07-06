# [실습]
# 시작!!!
# datasets.descibe()
# datasets.info()
# datasets.isnull().sum()

# pandas의  y 라벨의 종류가 무엇인지 확인하는 함수 쓸것
# numpy 에서는 np.unique(y,return_counts=True)

import numpy as np
import pandas as pd 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

#1.데이터

path = './_data/kaggle_titanic/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        )
test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       )
combine = [train_set,test_set]
print(train_set) # [891 rows x 11 columns]
# print(train_set.describe())
# print(train_set.info())
# train_set.describe()
# print(train_set.describe(include=['O']))
train_set[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# # 열(col)을 생존 여부로 나눔
# g = sns.FacetGrid(train_set, col='Survived')
# # 히스토그램으로 시각화, 연령의 분포를 확인, 히스토그램 bin을 20개로 설정
# g.map(plt.hist, 'Age', bins=20)

# grid = sns.FacetGrid(train_set, col='Survived', row='Pclass', hue="Pclass", height=2.2, aspect=1.6)

# grid.map(plt.hist, 'Age', alpha=.5, bins=20) # 투명도(alpha): 0.5

# # 범례 추가
# grid.add_legend();
# grid = sns.FacetGrid(train_set, row='Embarked', height=2.2, aspect=1.6)

# # Pointplot으로 시각화, x: 객실 등급, y: 생존 여부, 색깔: 성별, x축 순서: [1, 2, 3], 색깔 순서: [남성, 여성]
# grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep', order = [1, 2, 3], hue_order = ["male", "female"])

# grid.add_legend()
# grid = sns.FacetGrid(train_set, row='Embarked', col='Survived', height=2.2, aspect=1.6)

# # 바그래프로 시각화, x: 성별, y: 요금, Error bar: 표시 안 함
# grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None,order=["male","female"])

# grid.add_legend()
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
print(train_set.head())

drop_cols = ['Cabin','Ticket']
train_set.drop(drop_cols, axis = 1, inplace =True)
test_set.drop(drop_cols, axis = 1, inplace =True)

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_set['Title'], train_set['Sex'])
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_set[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

a = train_set.head()
train_set = train_set.drop(['Name', 'PassengerId'], axis=1)
test_set = test_set.drop(['Name'], axis=1)
combine = [train_set, test_set]

print(train_set.shape, test_set.shape)# (891, 9) (418, 9)
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_set.head()
grid = sns.FacetGrid(train_set, row='Pclass', col='Sex', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()

guess_ages = np.zeros((2,3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()
            # 위에서 guess_ages사이즈를 [2,3]으로 잡아뒀으므로 j의 범위도 이를 따름
            
            age_guess = guess_df.median()

            # age의 random값의 소수점을 .5에 가깝도록 변형
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)


print(train_set.isnull().sum())

train_set['AgeBand'] = pd.cut(train_set['Age'], 5)
# 임의로 5개 그룹을 지정
train_set[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
'''
test_set['Fare'].fillna(test_set['Fare'].dropna().median(), inplace=True)
print(test_set.isnull().sum())

print(train_set['Embarked'].value_counts())
train_set['Embarked'].fillna('S', inplace=True)
# print('Oldereest Passenger was of : ', train_set['Age'].max(), 'Years')
# print('Younged Passenger was of : ', train_set['Age'].min(), 'Years')
# print('Average Passenger was of : ', train_set['Age'].mean(), 'Years')
train_set['Initial'] = 0
for i in train_set:
    train_set['Initial']=train_set.Name.str.extract('([A-Za-z]+)\.')
train_set.groupby('Initial')['Age'].mean()
train_set['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
                             ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],
                             inplace=True)
test_set['Initial'] = 0
for i in test_set:
    test_set['Initial']=test_set.Name.str.extract('([A-Za-z]+)\.')
test_set['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
                             ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],
                             inplace=True)    
# print(train_set.groupby('Initial')['Age'].mean())
train_set.loc[(train_set.Age.isnull()) & (train_set.Initial=='Mr'),'Age']=33
train_set.loc[(train_set.Age.isnull()) & (train_set.Initial=='Mrs'),'Age']=36
train_set.loc[(train_set.Age.isnull()) & (train_set.Initial=='Master'),'Age']=5
train_set.loc[(train_set.Age.isnull()) & (train_set.Initial=='Miss'),'Age']=22
train_set.loc[(train_set.Age.isnull()) & (train_set.Initial=='Other'),'Age']=46
###############################
test_set.loc[(test_set.Age.isnull()) & (test_set.Initial=='Mr'),'Age']=33
test_set.loc[(test_set.Age.isnull()) & (test_set.Initial=='Mrs'),'Age']=36
test_set.loc[(test_set.Age.isnull()) & (test_set.Initial=='Master'),'Age']=5
test_set.loc[(test_set.Age.isnull()) & (test_set.Initial=='Miss'),'Age']=22
test_set.loc[(test_set.Age.isnull()) & (test_set.Initial=='Other'),'Age']=46
train_set['Age_band'] = 0
train_set.loc[train_set['Age'] <= 16, 'Age_band'] = 0
train_set.loc[(train_set['Age'] > 16) & (train_set['Age'] <= 32), 'Age_band'] = 1
train_set.loc[(train_set['Age'] > 32) & (train_set['Age'] <= 48), 'Age_band'] = 2
train_set.loc[(train_set['Age'] > 48) & (train_set['Age'] <= 64), 'Age_band'] = 3
train_set.loc[train_set['Age'] > 64, 'Age_band'] = 4
test_set['Age_band'] = 0
test_set.loc[test_set['Age'] <= 16, 'Age_band'] = 0
test_set.loc[(test_set['Age'] > 16) & (test_set['Age'] <= 32), 'Age_band'] = 1
test_set.loc[(test_set['Age'] > 32) & (test_set['Age'] <= 48), 'Age_band'] = 2
test_set.loc[(test_set['Age'] > 48) & (test_set['Age'] <= 64), 'Age_band'] = 3
test_set.loc[test_set['Age'] > 64, 'Age_band'] = 4

print(train_set['Age_band']) 
print(train_set.isnull().sum())
print(train_set.shape)# (891,11)


drop_cols = ['Name','Age']
train_set.drop(drop_cols, axis = 1, inplace =True)
test_set.drop(drop_cols, axis = 1, inplace =True)

cols = ['Sex','Embarked','Initial']
for col in cols:
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])
x = train_set.drop(['Survived'],axis=1) #axis는 컬럼 

print(x) #(891, 8)
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



x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.75,shuffle=True ,random_state=100)
#셔플을 False 할 경우 순차적으로 스플릿하다보니 훈련에서는 나오지 않는 값이 생겨 정확도가 떨어진다.
#디폴트 값인  shuffle=True 를 통해 정확도를 올린다.

#2. 모델 구성

model = Sequential()
model.add(Dense(100,input_dim=8))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#다중 분류로 나오는 아웃풋 노드의 개수는 y 값의 클래스의 수와 같다.활성화함수 'softmax'를 통해 
# 아웃풋의 합은 1이 된다.


#3. 컴파일,훈련
earlyStopping = EarlyStopping(monitor='loss', patience = 150, mod
+6
                              verbose=1,restore_best_weights=True
                              +6
model.compile(loss='binary_crossentropy', optimizer='adam', metri
+6
model.fit(x_train, y_train, epochs=500530, batch_size=8, 
                validation_split=0.3,
                callbacks = [earlyStopping],
                verbose=2
                )

#다중 분류 모델은 'categorical_crossentropy'만 사용한다 !!!!

#4.  평가,예측

loss = model.evaluate(x_test,y_test)
print('loss :',loss)
# print('accuracy :',acc)
# print("+++++++++  y_test       +++++++++")
# print(y_test[:5])
# print("+++++++++  y_pred     +++++++++++++")
# result = model.evaluate(x_test,y_test) 위에와 같은 개념 [0] 또는 [1]을 통해 출력가능
# print('loss :',result[0])
# print('accuracy :',result[1])




y_predict = model.predict(x_test)
y_predict[(y_predict<0.5)] = 0  
y_predict[(y_predict>=0.5)] = 1  
print(y_predict) 
print(y_test.shape) #(134,)


# y_test = np.argmax(y_test,axis=1)
# import tensorflow as tf
# y_test = np.argmax(y_test,axis=1)
# y_predict = np.argmax(y_predict,axis=1)
#pandas 에서 인코딩 진행시 argmax는 tensorflow 에서 임포트한다.
# print(y_test.shape) #(87152,7)
# y_test와 y_predict의  shape가 일치해야한다.



acc = accuracy_score(y_test, y_predict)
print('acc 스코어 :', acc)

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
y_summit = model.predict(test_set)

gender_submission['Survived'] = y_summit
submission = gender_submission.fillna(gender_submission.mean())
submission [(submission <0.5)] = 0  
submission [(submission >=0.5)] = 1  
submission = submission.astype(int)
submission.to_csv('test21.csv',index=True)

# loss : [1.284850001335144, 0.8206278085708618]
# acc 스코어 : 0.820627802690583

# 확인
'''
