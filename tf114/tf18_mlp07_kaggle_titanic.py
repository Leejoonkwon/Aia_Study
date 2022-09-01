import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score,accuracy_score


#1.데이터

path = 'D:\study_data\_data\_csv\kaggle_titanic/' # ".은 현재 폴더"
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

# train_set.head()
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
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
tf.compat.v1.disable_v2_behavior()
y = pd.get_dummies(y)
# print(x.shape,y.shape) #(891, 7) (891,)

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.75,shuffle=True,random_state=123)
# print(x_train.dtype,y_train.dtype) #float64 int32
# print(type(x_train),type(y_train)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
'''
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
'''
x = tf.compat.v1.placeholder(tf.float32,shape = [None,7])
y = tf.compat.v1.placeholder(tf.float32,shape = [None,2])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([7,100]))
b1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100]))

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100,100]))
b2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100]))

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100,100]))
b3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100]))

w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100,1]))
b4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
# hidden_layer 1
hidden_layer1 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(x, w1) + b1)
hidden_layer2 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(hidden_layer1, w2) + b2)
hidden_layer3 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(hidden_layer2, w3) + b3)
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(hidden_layer3, w4) + b4)
# hypothesis = tf.nn.softmax(tf.add(tf.matmul(x, w), b))

# model.add(Dense(1,activation='sigmoid',input_dim=2))
#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis-y)) #mse

loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) 
#binary_crossentropy
# model.compile(loss='binary_crossentropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1E-3)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epoch = 500
import time
start_time = time.time()
for epochs in range(epoch):
    cost_val,h_val,_ = sess.run([loss,hypothesis,train],
                                           feed_dict={x:x_train,y:y_train})
    if epochs %10 == 0 :
        print(epochs,'\t',"loss :",cost_val,'\n',h_val)    
        # print(epochs,'\t',"loss :",cost_val)    
   
##################################### [실습]   R2로 맹그러봐

y_predict = sess.run(tf.cast(h_val>=0.5,dtype=tf.float32))
from sklearn.metrics import r2_score,accuracy_score
import numpy as np
# h_val =abs(h_val)
# h_val = np.round(h_val,0)
print(y_predict)
y_test = y_test.values

acc = accuracy_score(y_train,y_predict)
end_time = time.time()-start_time
print('acc :', acc)
print('걸린 시간 :',end_time)
sess.close()   

# acc : 0.7035928143712575
# 걸린 시간 : 3.3510353565216064
