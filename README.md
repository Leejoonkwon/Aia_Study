# Titanic 분석

=======
## 데이터 확인

![image](https://user-images.githubusercontent.com/107663853/177349581-f3de1da1-04a0-49b8-b485-8dddfe67792a.png)

![image](https://user-images.githubusercontent.com/107663853/177350370-4343eb2d-11ac-486a-92ac-1deb4e3552db.png)

#print(train_set.info()) 코드로 확인해본 결과 탑승객 891명에 대한 데이터가 있는 것을 알 수 있으며 Age와 Cabin,Embarked의 열에

결측값이 있는 것을 알 수 있다. 추가로 dtypes: float64(2), int64(5), object(5)에서 소수가 포함된 숫자인 float64와 정수인 int64는

연산을 할 수 있지만 문자가 포함된 object는 숫자로 변환해주어야 연산이 가능하다.

### ** 데이터 분석 **

#### print(train_set.describe())

![image](https://user-images.githubusercontent.com/107663853/177352890-5e381f49-a375-40f1-b88d-65f68fd303fd.png)

float64(2), int64(5)형 데이터---- 

총 891명의 탑승객의 생존율은 38.4%(mean = 0.3838) 

#### print(train_set.describe(include=['O']))

![image](https://user-images.githubusercontent.com/107663853/177353895-5ff60b46-4c6a-466f-bd7f-64e955f0f271.png)

object형 데이터 ----- 

훈련 자료 남성 수 : 577명 (top의 Sex = male, freq의 Sex = 577)
훈련 자료 가장 많은 승선지 : S, 644명 (top의 Embarked = S, freq의 Emabarked =644)


#groupby에 as_index를 False로 하면 Pclass를 index로 사용하지 않음
#### ascending : 오름차순
#### as_index를 True로 하면 Pclass를 index로 사용


#### 훈련 자료에서 객실 등급(Pclass)에 따른 생존율 비교
# train_set[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#groupby에 as_index를 False로 하면 Pclass를 index로 사용하지 않음
#ascending : 오름차순
#as_index를 True로 하면 Pclass를 index로 사용

![image](https://user-images.githubusercontent.com/107663853/177355250-4788986e-1126-4064-8967-0244c5ee2641.png)

객실 등급이 좋을 수록 생존율이 높음

#### 훈련 자료에서 성별(Sex)에 따른 생존율 비교
# train_test[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

![image](https://user-images.githubusercontent.com/107663853/177355549-a6530720-08e0-455d-baff-e74450299194.png)

여성의 생존율이 남성보다 높음

#### 훈련 자료에서 함께 승선한 형제자매와 배우자 수(SibSp)에 따른 생존율 비교
# train_set[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

![image](https://user-images.githubusercontent.com/107663853/177356522-23accca6-c488-49b0-a8e5-6fc056056f32.png)

#### 훈련 자료에서 함께 승선한 부모와 자식 수(Parch)에 따른 생존율 비교
# train_set[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

![image](https://user-images.githubusercontent.com/107663853/177356913-4163e006-2307-4b99-9916-a98eeb6c0b5e.png)

동행이 적은 경우, 생존율이 높음

#### 훈련 자료에서 생존 여부(Survived)에 따른 연령(Age) 분포
# # 열(col)을 생존 여부로 나눔
g = sns.FacetGrid(train_set, col='Survived')
# 히스토그램으로 시각화, 연령의 분포를 확인, 히스토그램 bin을 20개로 설정
g.map(plt.hist, 'Age', bins=20)
plt.show()

![image](https://user-images.githubusercontent.com/107663853/177362659-fae38cdb-0577-49f8-b3b4-e8afa69d7281.png)

4세 이하의 유아의 생존율이 높음
15 ~ 25세 승객들의 생존율이 높음

#### 훈련 자료에서 객실 등급(Pclass)과 생존 여부(Survived)에 따른 연령(Age) 분포
grid = sns.FacetGrid(train_set, col='Survived', row='Pclass', hue="Pclass", height=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20) # 투명도(alpha): 0.5

# 범례 추가
grid.add_legend();
plt.show()

![image](https://user-images.githubusercontent.com/107663853/177363053-d6a24577-a850-4571-afe2-e32ed7aa1684.png)


객실 등급이 3등급인 경우, 승객 수는 가장 많고, 생존율도 가장 낮음
객실 등급이 2등급인 유아는 대부분 생존함
객실 등급이 1등급인 경우 생존율이 비교적 높음

#### 훈련자료에서 승선지(Embarked)와 객실 등급(Pclass)에 따른 생존율(Survived)
grid = sns.FacetGrid(train_set, row='Embarked', height=2.2, aspect=1.6)

# Pointplot으로 시각화, x: 객실 등급, y: 생존 여부, 색깔: 성별, x축 순서: [1, 2, 3], 색깔 순서: [남성, 여성]
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep', order = [1, 2, 3], hue_order = ["male", "female"])

grid.add_legend()
plt.show()

![image](https://user-images.githubusercontent.com/107663853/177363509-2254cd4b-4032-43eb-baad-65b4577da3c3.png)

승선지가 C와 Q인 경우, 남성의 티켓 등급이 3등급일 때 2등급보다 생존율이 높을 가능성이 있음

#### 훈련 자료에서 승선지(Embarked), 생존 여부(Survived), 성별(Sex)에 따른 요금(Fare)
grid = sns.FacetGrid(train_set, row='Embarked', col='Survived', height=2.2, aspect=1.6)

# 바그래프로 시각화, x: 성별, y: 요금, Error bar: 표시 안 함
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None,order=["male","female"])

grid.add_legend()
plt.show()

![image](https://user-images.githubusercontent.com/107663853/177363789-ee80d323-24d6-4078-96d8-a9786c71d986.png)

승선지가 S또는 C인 경우, 생존한 승객들의 평균 요금이 비교적 높음

### 데이터 전처리

#### 안쓸 변수(Ticket, Cabin) 제거

drop_cols = ['Cabin','Ticket']
train_set.drop(drop_cols, axis = 1, inplace =True)
test_set.drop(drop_cols, axis = 1, inplace =True)

#### 이름으로 성별 분석

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_set['Title'], train_set['Sex'])

![image](https://user-images.githubusercontent.com/107663853/177366548-8c5d4c50-6519-4397-9fd5-637038b281aa.png)

female에서는 Miss와 Mrs가, male에서는 Master와 Mr가 두드러지게 나타남(Mlle와 Ms 는 Miss의, Ms는 Mrs의 불어식 표현)

나머지는 Rare로 분류

![image](https://user-images.githubusercontent.com/107663853/177366917-8233d37c-797c-407d-a517-0cd05f1185bb.png)

#### Title 변수를 숫자형 변수로 바꿔줌

![image](https://user-images.githubusercontent.com/107663853/177367300-c19d3096-0e79-45af-a1f5-25894292ba67.png)

###### 안쓸 변수(Name, PassengerId) 제거
train_set = train_set.drop(['Name', 'PassengerId'], axis=1)
test_set = test_set.drop(['Name'], axis=1)
combine = [train_set, test_set]

print(train_set.shape, test_set.shape)# (891, 9) (418, 9)

###### 성별(Sex) 변수를 숫자 범주형 변수로 바꿔줌
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


![image](https://user-images.githubusercontent.com/107663853/177369124-c809986c-60dc-4aec-979e-b69e788ce304.png)

###### 객실 등급(Pclass)과 성별(Sex) 시각화
grid = sns.FacetGrid(train_set, row='Pclass', col='Sex', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()

![image](https://user-images.githubusercontent.com/107663853/177369480-72f009e3-0945-4590-9dde-fc4c8fdf2e2e.png)

Pclass, Sex와 Age와의 결합이 2열 3행으로 구성됨

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

![image](https://user-images.githubusercontent.com/107663853/177369873-918fc280-00f5-46e9-ac96-dc67721f9e1a.png)

age의 결측치가 채워진 것을 볼 수 있다.
#### 연령(Age) 변수를 범주형 변수로 바꿔줌

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

![image](https://user-images.githubusercontent.com/107663853/177437054-216d2d9b-bc2b-4b71-831c-63f7e9118fc1.png)

#### SibSp와 Parch를 가족과의 동반여부를 알 수 있는 새로운 변수로 통합

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_set[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

![image](https://user-images.githubusercontent.com/107663853/177437285-29e9c8cc-40c3-42c6-86e9-2aaa61bd9abe.png)

FamilySize가 1인 것은 가족과 동반하지 않음을 의미

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_set[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

![image](https://user-images.githubusercontent.com/107663853/177437434-70b1d91f-bee5-4152-be40-b9a59571f998.png)

1은 동반X
0은 동반했다는 새로운 변수 IsAlone을 생성

train_set = train_set.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_set = test_set.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_set, test_set]
print(train_set.head())

![image](https://user-images.githubusercontent.com/107663853/177437607-bb5f66a7-9b92-4070-9804-7e2649a47698.png)

'Parch', 'SibSp', 'FamilySize' 특성들을 결합해 IsAlone이라는 특성을 만들었으니 3가지 특성은 삭제 해준다.

test_set 에 Fare 결측치는 중앙값으로 대체 후 결측 여부 확인

test_set['Fare'].fillna(test_set['Fare'].dropna().median(), inplace=True)
print(test_set.isnull().sum())

![image](https://user-images.githubusercontent.com/107663853/177437968-f2b6c736-4536-4165-9b76-a871ba0e1c83.png)

Embarked의 결측치는 valuse_counts를 통해 가장 많은 값인 'S'로 대치한다.

print(train_set['Embarked'].value_counts())

train_set['Embarked'].fillna('S', inplace=True)

![image](https://user-images.githubusercontent.com/107663853/177438050-c7d666d5-a88e-438d-91ba-1daf56259899.png)

승선지(Ebmarked) 변수를 범주형 변수로 바꿔줌
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

a = train_set.head()
print(a)

![image](https://user-images.githubusercontent.com/107663853/177438520-a8dc8dc7-a14a-41d0-b234-3736ef8867a9.png)

요금(Fare)을 숫자 범주형 변수로 바꿔줌

train_set['FareBand'] = pd.qcut(train_set['Fare'], 4)
a = train_set[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
print(a)

![image](https://user-images.githubusercontent.com/107663853/177438677-79a83639-052e-45d3-a0f7-4568f469b249.png)
