from time import time
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np 

#1. 데이터
path = 'D:\study_data\_data\_csv\dacon_travel/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)


train_set['TypeofContact'].fillna('Self Enquiry', inplace=True)
test_set['TypeofContact'].fillna('Self Enquiry', inplace=True)
train_set['Age'].fillna(train_set.groupby('Designation')['Age'].transform('mean'), inplace=True)
test_set['Age'].fillna(test_set.groupby('Designation')['Age'].transform('mean'), inplace=True)
train_set['Age']=np.round(train_set['Age'],0).astype(int)
test_set['Age']=np.round(test_set['Age'],0).astype(int)



train_set['MonthlyIncome'].fillna(train_set.groupby('Designation')['MonthlyIncome'].transform('mean'), inplace=True)
test_set['MonthlyIncome'].fillna(test_set.groupby('Designation')['MonthlyIncome'].transform('mean'), inplace=True)
print(train_set.describe) #(1955, 19)
print(train_set[train_set['MonthlyIncome'].notnull()].groupby(['Designation'])['MonthlyIncome'].mean())

train_set['NumberOfChildrenVisiting'].fillna(train_set.groupby('MaritalStatus')['NumberOfChildrenVisiting'].transform('mean'), inplace=True)
test_set['NumberOfChildrenVisiting'].fillna(test_set.groupby('MaritalStatus')['NumberOfChildrenVisiting'].transform('mean'), inplace=True)
train_set['NumberOfFollowups'].fillna(train_set.groupby('NumberOfChildrenVisiting')['NumberOfFollowups'].transform('mean'), inplace=True)
test_set['NumberOfFollowups'].fillna(test_set.groupby('NumberOfChildrenVisiting')['NumberOfFollowups'].transform('mean'), inplace=True)
# combine = [train_set,test_set]
# for dataset in combine:    
#     dataset.loc[ dataset['NumberOfChildrenVisiting'] < 1, 'NumberOfChildrenVisiting'] = 0
#     dataset.loc[ dataset['NumberOfChildrenVisiting'] >= 1, 'NumberOfChildrenVisiting'] = 1
# print(train_set[train_set['DurationOfPitch'].notnull()].groupby(['NumberOfChildrenVisiting'])['DurationOfPitch'].mean())
# print(train_set.isnull().sum()) 

train_set['DurationOfPitch']=train_set['DurationOfPitch'].fillna(0)
test_set['DurationOfPitch']=test_set['DurationOfPitch'].fillna(0)
# train_set['DurationOfPitch'].fillna(train_set.groupby('NumberOfChildrenVisiting')['DurationOfPitch'].transform('mean'), inplace=True)
# test_set['DurationOfPitch'].fillna(test_set.groupby('NumberOfChildrenVisiting')['DurationOfPitch'].transform('mean'), inplace=True)
# print(train_set.isnull().sum()) 


print(train_set[train_set['DurationOfPitch'].notnull()].groupby(['NumberOfChildrenVisiting'])['DurationOfPitch'].mean())


train_set['PreferredPropertyStar'].fillna(train_set.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)
test_set['PreferredPropertyStar'].fillna(test_set.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)
print(train_set[train_set['PreferredPropertyStar'].notnull()].groupby(['ProdTaken'])['PreferredPropertyStar'].mean())


combine = [train_set,test_set]
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 20, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 29), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 29) & (dataset['Age'] <= 39), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 39) & (dataset['Age'] <= 49), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 49) & (dataset['Age'] <= 59), 'Age'] = 4
    dataset.loc[ dataset['Age'] > 59, 'Age'] = 5

# Trial 4 finished with value: 1.0 and
# parameters: {'n_estimators': 2795, 
#              'depth': 10, 
#              'fold_permutation_block': 58,
#              'od_pval': 0.05209401437557892, 
# 'l2_leaf_reg': 0.8427440855520927}. Best is trial 0 with value: 1.0.    
# train_set = train_set.drop(['AgeBand'], axis=1)
# print(train_set[train_set['NumberOfTrips'].notnull()].groupby(['DurationOfPitch'])['PreferredPropertyStar'].mean())
train_set['NumberOfTrips'].fillna(train_set.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)
test_set['NumberOfTrips'].fillna(test_set.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)
# print(train_set[train_set['NumberOfChildrenVisiting'].notnull()].groupby(['MaritalStatus'])['NumberOfChildrenVisiting'].mean())

# print(train_set['Occupation'].unique()) # ['Small Business' 'Salaried' 'Large Business' 'Free Lancer']
train_set.loc[ train_set['Occupation'] =='Free Lancer' , 'Occupation'] = 'Salaried'
test_set.loc[ test_set['Occupation'] =='Free Lancer' , 'Occupation'] = 'Salaried'

train_set.loc[ train_set['Gender'] =='Fe Male' , 'Gender'] = 'Female'
test_set.loc[ test_set['Gender'] =='Fe Male' , 'Gender'] = 'Female'
cols = ['TypeofContact','Occupation','Gender','ProductPitched','MaritalStatus','Designation']
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook

for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])

# print(train_set)

# print(train_set['TypeofContact'])
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
                     
                           
# print(train_set['Designation'].unique())

# Age_out_index= outliers(train_set['Age'])[0]
# TypeofContact_out_index= outliers(train_set['TypeofContact'])[0] # 0
# CityTier_out_index= outliers(train_set['CityTier'])[0] # 0
DurationOfPitch_out_index= outliers(train_set['DurationOfPitch'])[0] #44
Gender_out_index= outliers(train_set['Gender'])[0] # 0
NumberOfPersonVisiting_out_index= outliers(train_set['NumberOfPersonVisiting'])[0] # 1
NumberOfFollowups_out_index= outliers(train_set['NumberOfFollowups'])[0] # 0
ProductPitched_index= outliers(train_set['ProductPitched'])[0] # 0
PreferredPropertyStar_out_index= outliers(train_set['PreferredPropertyStar'])[0]  # 0
MaritalStatus_out_index= outliers(train_set['MaritalStatus'])[0] # 0
NumberOfTrips_out_index= outliers(train_set['NumberOfTrips'])[0] # 38
Passport_out_index= outliers(train_set['Passport'])[0] # 0
PitchSatisfactionScore_out_index= outliers(train_set['PitchSatisfactionScore'])[0] # 0
OwnCar_out_index= outliers(train_set['OwnCar'])[0] # 0
NumberOfChildrenVisiting_out_index= outliers(train_set['NumberOfChildrenVisiting'])[0] # 0
Designation_out_index= outliers(train_set['Designation'])[0] # 89
MonthlyIncome_out_index= outliers(train_set['MonthlyIncome'])[0] # 138

lead_outlier_index = np.concatenate((#Age_out_index,                            # acc : 0.8650306748466258
                                    #  TypeofContact_out_index,                 # acc : 0.8920454545454546
                                    #  CityTier_out_index,                      # acc : 0.8920454545454546
                                     DurationOfPitch_out_index,               # acc : 0.9156976744186046
                                    #  Gender_out_index,                        # acc : 0.8920454545454546
                                    #  NumberOfPersonVisiting_out_index,        # acc : 0.8835227272727273
                                    #  NumberOfFollowups_out_index,             # acc : 0.8942598187311178
                                    #  ProductPitched_index,                    # acc : 0.8920454545454546
                                    #  PreferredPropertyStar_out_index,         # acc : 0.8920454545454546
                                    #  MaritalStatus_out_index,                 # acc : 0.8920454545454546
                                    #  NumberOfTrips_out_index,                 # acc : 0.8670520231213873
                                    #  Passport_out_index,                      # acc : 0.8920454545454546
                                    #  PitchSatisfactionScore_out_index,        # acc : 0.8920454545454546
                                    #  OwnCar_out_index,                        # acc : 0.8920454545454546
                                    #  NumberOfChildrenVisiting_out_index,      # acc : 0.8920454545454546
                                    #  Designation_out_index,                   # acc : 0.8869047619047619
                                    #  MonthlyIncome_out_index                  # acc : 0.8932926829268293
                                     ),axis=None)
print(len(lead_outlier_index)) #577

lead_not_outlier_index = []
for i in train_set.index:
    if i not in lead_outlier_index :
        lead_not_outlier_index.append(i)
train_set_clean = train_set.loc[lead_not_outlier_index]      
train_set_clean = train_set_clean.reset_index(drop=True)
# print(train_set_clean)
x = train_set_clean.drop(['ProdTaken',
                          'NumberOfChildrenVisiting',
                          'NumberOfPersonVisiting',
                          'OwnCar', 
                          'MonthlyIncome', 
                          'NumberOfFollowups',
                          'NumberOfTrips',
                          ], axis=1)
# x = train_set_clean.drop(['ProdTaken'], axis=1)
test_set = test_set.drop(['NumberOfChildrenVisiting',
                          'NumberOfPersonVisiting',
                          'OwnCar', 
                          'MonthlyIncome', 
                          'NumberOfFollowups',
                          'NumberOfTrips',
                          ], axis=1)
y = train_set_clean['ProdTaken']
print(x.shape) #1911,13

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.91,shuffle=True,random_state=1234,stratify=y)

from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
                                         
                                 
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error
# XGB 하이퍼 파라미터들 값 지정
 
'''
optuna.trial.Trial.suggest_categorical() : 리스트 범위 내에서 값을 선택한다.
optuna.trial.Trial.suggest_int() : 범위 내에서 정수형 값을 선택한다.
optuna.trial.Trial.suggest_float() : 범위 내에서 소수형 값을 선택한다.
optuna.trial.Trial.suggest_uniform() : 범위 내에서 균일분포 값을 선택한다.
optuna.trial.Trial.suggest_discrete_uniform() : 범위 내에서 이산 균일분포 값을 선택한다.
optuna.trial.Trial.suggest_loguniform() : 범위 내에서 로그 함수 값을 선택한다.
'''
# learning_rate : float range: (0,1]
# depth : int, [default=6]   range: [1,+inf]
# od_pval : float, [default=None] range: [0,1]
# model_size_reg : float, [default=None] range: [0,+inf]
# l2_leaf_reg : float, [default=3.0]  range: [0,+inf]
#  fold_permutation_block : int, [default=1] T[1, 256]. 
def objectiveCAT(trial: Trial, x_train, y_train, x_test):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 500, 4000),
        'depth' : trial.suggest_int('depth', 8, 16),
        'fold_permutation_block' : trial.suggest_int('fold_permutation_block', 1, 256),
        'learning_rate' : trial.suggest_float('learning_rate', 0, 1),
        'od_pval' : trial.suggest_float('od_pval', 0, 1),
        'l2_leaf_reg' : trial.suggest_float('l2_leaf_reg', 0, 4),
        'random_state' :trial.suggest_int('random_state', 1, 2000)
    }
    
    # 학습 모델 생성
    model = CatBoostClassifier(**param)
    CAT_model = model.fit(x_train, y_train, verbose=True) # 학습 진행
    
    # 모델 성능 확인
    score = accuracy_score(CAT_model.predict(x_test), y_test)
    
    return score
# MAE가 최소가 되는 방향으로 학습을 진행
# TPESampler : Sampler using TPE (Tree-structured Parzen Estimator) algorithm.
study = optuna.create_study(direction='maximize', sampler=TPESampler())

# n_trials 지정해주지 않으면, 무한 반복
study.optimize(lambda trial : objectiveCAT(trial, x, y, x_test), n_trials = 5)

print('Best trial : score {}, \nparams {}'.format(study.best_trial.value, study.best_trial.params))

# # 하이퍼파라미터별 중요도를 확인할 수 있는 그래프
# print(optuna.visualization.plot_param_importances(study))

# # 하이퍼파라미터 최적화 과정을 확인
# optuna.visualization.plot_optimization_history(study)

# plt.show()