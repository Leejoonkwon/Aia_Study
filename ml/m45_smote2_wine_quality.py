# 실습!! 시작 !!!
# csv로 맹그러!!!
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import StandardScaler,MinMaxScaler 
from xgboost import XGBClassifier,XGBRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.over_sampling import SMOTE

#1. 데이터
path = 'C:\_data\wine/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'winequality-white.csv',
                        sep=';',index_col=None,header=0)
print(train_set.describe())

# print(train_set.isnull().sum()) #(4898, 12)
def outliers(data_out):
    quartile_1, q2 , quartile_3 = np.percentile(data_out,
                                               [25,50,75]) # percentile 백분위
    print("1사분위 : ",quartile_1) # 25% 위치인수를 기점으로 사이에 값을 구함
    print("q2 : ",q2) # 50% median과 동일 
    print("3사분위 : ",quartile_3) # 75% 위치인수를 기점으로 사이에 값을 구함
    iqr =quartile_3-quartile_1  # 75% -25%
    print("iqr :" ,iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound)|
                    (data_out<lower_bound))
fixed_acidity_out_index= outliers(train_set['fixed acidity'])[0]
volatile_acidity_out_index= outliers(train_set['volatile acidity'])[0]
citric_acid_out_index= outliers(train_set['citric acid'])[0]
residual_sugar_out_index= outliers(train_set['residual sugar'])[0]
chlorides_out_index= outliers(train_set['chlorides'])[0]
free_sulfur_dioxide_out_index= outliers(train_set['free sulfur dioxide'])[0]
total_sulfur_dioxide_out_index= outliers(train_set['total sulfur dioxide'])[0]
density_out_index= outliers(train_set['density'])[0]
pH_out_index= outliers(train_set['pH'])[0]
sulphates_out_index= outliers(train_set['sulphates'])[0]
alcohol_out_index= outliers(train_set['alcohol'])[0]

lead_outlier_index = np.concatenate((fixed_acidity_out_index,
                                     volatile_acidity_out_index,
                                     citric_acid_out_index,
                                     residual_sugar_out_index,
                                     chlorides_out_index,
                                     free_sulfur_dioxide_out_index,
                                     total_sulfur_dioxide_out_index,
                                     density_out_index,
                                     pH_out_index,
                                     sulphates_out_index,
                                     sulphates_out_index,
                                     alcohol_out_index,
                                   
                                     ),axis=None)
print(len(lead_outlier_index)) #577
# print(lead_outlier_index)

lead_not_outlier_index = []
for i in train_set.index:
    if i not in lead_outlier_index :
        lead_not_outlier_index.append(i)
train_set_clean = train_set.loc[lead_not_outlier_index]      
train_set_clean = train_set_clean.reset_index(drop=True)

##############Pandas Dataframe을 Numpy로 바꾸기 
# .values 또는 .to_numpy() 를 사용해 numpy 배열로 변환
# train_set=train_set.to_numpy()
# train_set=train_set.values
# print(type(train_set)) #<class 'numpy.ndarray'>

x = train_set.drop(['quality'],axis=1)
y = train_set['quality']
# x = train_set[:,:11]
# y = train_set[:,11]
print(x.shape,y.shape) # (4898, 11) (4898,)

# print(np.unique(y,return_counts=True))
# (array([3, 4, 5, 6, 7, 8, 9], dtype=int64), 
#  array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
# pandas 타입이라면 df['확인할 컬럼'].value_counts()로 확인 가능 
# from tensorflow.keras.utils import to_categorical 
# y = to_categorical(y)
# print(x.shape)
# print(y.shape)
from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    shuffle=True, random_state=123, 
                                                    train_size=0.85,stratify=y)
from sklearn.preprocessing import MinMaxScaler,StandardScaler
print(pd.Series(y_train).value_counts()) # 데이터 증폭 
# 6    1758
# 5    1166
# 7     704
# 8     140
# 4     130
# 3      16
# 9       4
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
smote = SMOTE(random_state=123,k_neighbors=3)
x_train,y_train = smote.fit_resample(x_train,y_train)

print(pd.Series(y_train).value_counts()) # 데이터 증폭 
# 4    1758
# 5    1758
# 6    1758
# 7    1758
# 8    1758
# 3    1758
# 9    1758

#2. 모델
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)

score = model.score(x_test,y_test)
from sklearn.metrics import accuracy_score,f1_score
print('acc_score :',accuracy_score(y_test,y_predict))
print('f1_score(macro) :',f1_score(y_test,y_predict,average='macro'))
# f1_score(macro) : 0.4397558777039733 이진 분류일 때 사용
# print('f1_score(micro) :',f1_score(y_test,y_predict,average='micro'))
# f1_score(micro) : 0.7163265306122448 다중 분류일 때 사용

# acc_score : 0.6802721088435374
# f1_score(macro) : 0.42765507578677014





