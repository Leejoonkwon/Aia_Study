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
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False

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
train_set = train_set.drop(['Name','PassengerId'], axis=1)
test_set = test_set.drop(['Name'], axis=1)
combine = [train_set, test_set]

print(train_set.shape, test_set.shape)# (891, 9) (418, 9)
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_set.head()
grid = sns.FacetGrid(train_set, row='Pclass', col='Sex', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

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
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
train_set = train_set.drop(['AgeBand'], axis=1)
combine = [train_set, test_set]
train_set.head()

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_set[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_set[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

train_set = train_set.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_set = test_set.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_set, test_set]
print(train_set.head())



test_set['Fare'].fillna(test_set['Fare'].dropna().median(), inplace=True)
print(test_set.isnull().sum())

print(train_set['Embarked'].value_counts())

train_set['Embarked'].fillna('S', inplace=True)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_set.head()
train_set['FareBand'] = pd.qcut(train_set['Fare'], 4)
a = train_set[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_set = train_set.drop(['FareBand'], axis=1)
combine = [train_set, test_set]
train_set.head(10)

# print('Oldereest Passenger was of : ', train_set['Age'].max(), 'Years')
# print('Younged Passenger was of : ', train_set['Age'].min(), 'Years')
# print('Average Passenger was of : ', train_set['Age'].mean(), 'Years')
x = train_set.drop(['Survived'],axis=1) #axis는 컬럼 

print(x) #(891, 7)
y = train_set['Survived']
print(y.shape) #(891,)
a = x.head()
print(a)



gender_submission = pd.read_csv(path + 'gender_submission.csv',#예측에서 쓸거야!!
                      )
#


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.75,shuffle=True ,random_state=100)
#셔플을 False 할 경우 순차적으로 스플릿하다보니 훈련에서는 나오지 않는 값이 생겨 정확도가 떨어진다.
#디폴트 값인  shuffle=True 를 통해 정확도를 올린다.

#2. 모델 구성

model = Sequential()
model.add(Dense(500,input_dim=7))
model.add(Dense(500, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#다중 분류로 나오는 아웃풋 노드의 개수는 y 값의 클래스의 수와 같다.활성화함수 'softmax'를 통해 
# 아웃풋의 합은 1이 된다.


#3. 컴파일,훈련
earlyStopping = EarlyStopping(monitor='loss', patience=100, mode='min', 
                              verbose=1,restore_best_weights=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=680, batch_size=80, 
                validation_split=0.25,
                callbacks = [earlyStopping],
                verbose=2)

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
y_predict = y_predict.astype(int)
print(y_predict) 
print(y_predict.shape) #(223,1)
print(y_test)
print(y_test.shape)# (223,)

acc = accuracy_score(y_test, y_predict)
print('acc 스코어 :', acc)
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'],marker='.',c='red',label='loss') #순차적으로 출력이므로  y값 지정 필요 x
plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss')
plt.grid()
plt.title('영어싫어') #맥플러립 한글 깨짐 현상 알아서 해결해라 
plt.ylabel('loss')
plt.xlabel('epochs')
# plt.legend(loc='upper right')
plt.legend()
plt.show()
# print(x_test) # (223,8)
# print(test_set) # (418,8)
# print(test_set) # (418,8)
test_set = test_set.drop(['PassengerId'], axis=1)
y_summit = model.predict(test_set)


gender_submission['Survived'] = y_summit
submission = gender_submission.fillna(gender_submission.mean())
submission = submission.astype(int)
submission.to_csv('test21.csv',index=True)

# loss : [0.9520659446716309, 0.8026905655860901]
# acc 스코어 : 0.5695067264573991

# 확인

