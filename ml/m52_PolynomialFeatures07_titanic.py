import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error


#1.데이터

path = 'C:\_data\kaggle_titanic/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        )
test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       )
combine = [train_set,test_set]
print(train_set) # [891 rows x 11 columns]

train_set[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)



print(test_set) # [418 rows x 10 columns]
print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계


# train_set = train_set.fillna(train_set.median())
print(test_set.isnull().sum())

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


x = train_set.drop(['Survived'],axis=1) #axis는 컬럼 

print(x) #(891, 7)
y = train_set['Survived']
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split,KFold 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier


x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8,random_state=1234,
)


#2. 모델
model = make_pipeline(StandardScaler(),
                      CatBoostClassifier(verbose=False))

#3. 훈련
model.fit(x_train,y_train)


#4. 평가,예측
kfold = KFold(n_splits=5,random_state=123,shuffle=True)
print('기냥 스코어 :',model.score(x_test,y_test))
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,x_train,y_train,cv=kfold,scoring='r2')
print("기냥 CV : ",scores)
print("기냥 CV 엔빵 : ",np.mean(scores))



################## PolynomialFeatures 후 

pf = PolynomialFeatures(degree=2,include_bias=False)
xp = pf.fit_transform(x)
print(xp.shape) #(506, 105)

x_train,x_test,y_train,y_test = train_test_split(
    xp,y,train_size=0.8,random_state=1234,
)

#2. 모델
model = make_pipeline(StandardScaler(),
                      CatBoostClassifier(verbose=False))

#3. 훈련
model.fit(x_train,y_train)


#4. 평가,예측

print('poly 스코어 :',model.score(x_test,y_test))
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,x_train,y_train,cv=kfold,scoring='r2')
print("폴리 CV : ",scores)
print("폴리 CV 엔빵 : ",np.mean(scores))
#==================gridsearch
# 최적의 매개변수 : RandomForestClassifier(max_depth=6, min_samples_leaf=3, n_jobs=-1)
# 최적의 파라미터 : {'max_depth': 6, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 100, 'n_jobs': -1}  
# best_score : 0.813158672313602
# model_score : 0.8268156424581006
# accuracy_score : 0.8268156424581006
# 최적 튠  ACC : 0.8268156424581006
# 걸린 시간 : 18.97 초

######Bagging 후 acc model xgb
# model.score : 0.8379888268156425

# 보팅 결과 : 0.838
# CatBoostClassifier 정확도 : 0.843575
# XGBClassifier 정확도 : 0.837989
# LGBMClassifier 정확도 : 0.843575
# RandomForestClassifier 정확도 : 0.837989

# 기냥 스코어 : 0.8268156424581006
# 기냥 CV :  [0.2955665  0.01896691 0.28330042 0.36282051 0.14493355]
# 기냥 CV 엔빵 :  0.22111757916784605
# (891, 35)
# poly 스코어 : 0.8324022346368715
# 폴리 CV :  [0.20751232 0.01896691 0.22097871 0.36282051 0.2333887 ]
# 폴리 CV 엔빵 :  0.20873343105481493



