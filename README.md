# Titanic 분석
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
