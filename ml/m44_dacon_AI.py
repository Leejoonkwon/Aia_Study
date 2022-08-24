from tabnanny import verbose
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
import pandas as pd
import numpy as np 
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

#1. 데이터
path = 'D:\study_data\_data\_csv\dacon_ai/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       ).drop(columns=['ID'])
# print(train_set.shape)  #(39607, 71)      

train_x = train_set.filter(regex='X') # Input : X Featrue : 56
train_y = train_set.filter(regex='Y') # Output : Y Feature : 14

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
    
print(train_x.info())


X_01_out_index= outliers(train_x['X_01'])[0]
X_02_out_index= outliers(train_x['X_02'])[0]
X_03_out_index= outliers(train_x['X_03'])[0]
X_04_out_index= outliers(train_x['X_04'])[0]
X_05_out_index= outliers(train_x['X_05'])[0]
X_06_out_index= outliers(train_x['X_06'])[0]
X_07_out_index= outliers(train_x['X_07'])[0]
X_08_out_index= outliers(train_x['X_08'])[0]
X_09_out_index= outliers(train_x['X_09'])[0]
X_10_out_index= outliers(train_x['X_10'])[0]
X_11_out_index= outliers(train_x['X_11'])[0]
X_12_out_index= outliers(train_x['X_12'])[0]
X_13_out_index= outliers(train_x['X_13'])[0]
X_14_out_index= outliers(train_x['X_14'])[0]
X_15_out_index= outliers(train_x['X_15'])[0]
X_16_out_index= outliers(train_x['X_16'])[0]
X_17_out_index= outliers(train_x['X_17'])[0]
X_18_out_index= outliers(train_x['X_18'])[0]
X_19_out_index= outliers(train_x['X_19'])[0]
X_20_out_index= outliers(train_x['X_20'])[0]
X_21_out_index= outliers(train_x['X_21'])[0]
X_22_out_index= outliers(train_x['X_22'])[0]
X_23_out_index= outliers(train_x['X_23'])[0]
X_24_out_index= outliers(train_x['X_24'])[0]
X_25_out_index= outliers(train_x['X_25'])[0]
X_26_out_index= outliers(train_x['X_26'])[0]
X_27_out_index= outliers(train_x['X_27'])[0]
X_28_out_index= outliers(train_x['X_28'])[0]
X_29_out_index= outliers(train_x['X_29'])[0]
X_30_out_index= outliers(train_x['X_30'])[0]
X_31_out_index= outliers(train_x['X_31'])[0]
X_32_out_index= outliers(train_x['X_32'])[0]
X_33_out_index= outliers(train_x['X_33'])[0]
X_34_out_index= outliers(train_x['X_34'])[0]
X_35_out_index= outliers(train_x['X_35'])[0]
X_36_out_index= outliers(train_x['X_36'])[0]
X_37_out_index= outliers(train_x['X_37'])[0]
X_38_out_index= outliers(train_x['X_38'])[0]
X_39_out_index= outliers(train_x['X_39'])[0]
X_40_out_index= outliers(train_x['X_40'])[0]
X_41_out_index= outliers(train_x['X_41'])[0]
X_42_out_index= outliers(train_x['X_42'])[0]
X_43_out_index= outliers(train_x['X_43'])[0]
X_44_out_index= outliers(train_x['X_44'])[0]
X_45_out_index= outliers(train_x['X_45'])[0]
X_46_out_index= outliers(train_x['X_46'])[0]
X_47_out_index= outliers(train_x['X_47'])[0]
X_48_out_index= outliers(train_x['X_48'])[0]
X_49_out_index= outliers(train_x['X_49'])[0]
X_50_out_index= outliers(train_x['X_50'])[0]
X_51_out_index= outliers(train_x['X_51'])[0]
X_52_out_index= outliers(train_x['X_52'])[0]
X_53_out_index= outliers(train_x['X_53'])[0]
X_54_out_index= outliers(train_x['X_54'])[0]
X_55_out_index= outliers(train_x['X_55'])[0]
X_56_out_index= outliers(train_x['X_56'])[0]


lead_outlier_index = np.concatenate((X_01_out_index,                 # acc : 0.8920454545454546
                                    #  X_02_out_index,                      # acc : 0.8920454545454546
                                    #  X_03_out_index,               # acc : 0.9156976744186046
                                    #  X_04_out_index,                        # acc : 0.8920454545454546
                                    #  X_05_out_index,        # acc : 0.8835227272727273
                                    #  X_06_out_index,             # acc : 0.8942598187311178
                                    #  X_07_out_index,                    # acc : 0.8920454545454546
                                    #  X_08_out_index,         # acc : 0.8920454545454546
                                    #  X_09_out_index,                 # acc : 0.8920454545454546
                                    #  X_10_out_index,                 # acc : 0.8670520231213873
                                    #  X_11_out_index,                      # acc : 0.8920454545454546
                                    #  X_12_out_index,        # acc : 0.8920454545454546
                                    #  X_13_out_index,                        # acc : 0.8920454545454546
                                    #  X_14_out_index,      # acc : 0.8920454545454546
                                    #  X_15_out_index,                   # acc : 0.8869047619047619
                                    #  X_16_out_index                  # acc : 0.8932926829268293
                                    #  X_17_out_index                  # acc : 0.8932926829268293
                                    #  X_18_out_index                  # acc : 0.8932926829268293
                                    #  X_19_out_index                  # acc : 0.8932926829268293
                                    #  X_20_out_index                  # acc : 0.8932926829268293
                                    #  X_21_out_index                  # acc : 0.8932926829268293
                                    #  X_22_out_index                  # acc : 0.8932926829268293
                                    #  X_23_out_index                  # acc : 0.8932926829268293
                                    #  X_24_out_index                  # acc : 0.8932926829268293
                                    #  X_25_out_index                  # acc : 0.8932926829268293
                                    #  X_26_out_index                  # acc : 0.8932926829268293
                                    #  X_27_out_index                  # acc : 0.8932926829268293
                                    #  X_28_out_index                  # acc : 0.8932926829268293
                                    #  X_29_out_index                  # acc : 0.8932926829268293
                                    #  X_30_out_index                  # acc : 0.8932926829268293
                                    #  X_31_out_index                  # acc : 0.8932926829268293
                                    #  X_32_out_index                  # acc : 0.8932926829268293
                                    #  X_33_out_index                  # acc : 0.8932926829268293
                                    #  X_34_out_index                  # acc : 0.8932926829268293
                                    #  X_35_out_index                  # acc : 0.8932926829268293
                                    #  X_36_out_index                  # acc : 0.8932926829268293
                                    #  X_37_out_index                  # acc : 0.8932926829268293
                                    #  X_38_out_index                  # acc : 0.8932926829268293
                                    #  X_39_out_index                  # acc : 0.8932926829268293
                                    #  X_40_out_index                  # acc : 0.8932926829268293
                                    #  X_41_out_index                  # acc : 0.8932926829268293
                                    #  X_42_out_index                  # acc : 0.8932926829268293
                                    #  X_43_out_index                  # acc : 0.8932926829268293
                                    #  X_44_out_index                  # acc : 0.8932926829268293
                                    #  X_45_out_index                  # acc : 0.8932926829268293
                                    #  X_46_out_index                  # acc : 0.8932926829268293
                                    #  X_47_out_index                  # acc : 0.8932926829268293
                                    #  X_48_out_index                  # acc : 0.8932926829268293
                                    #  X_49_out_index                  # acc : 0.8932926829268293
                                    #  X_50_out_index                  # acc : 0.8932926829268293
                                    #  X_51_out_index                  # acc : 0.8932926829268293
                                    #  X_52_out_index                  # acc : 0.8932926829268293
                                    #  X_53_out_index                  # acc : 0.8932926829268293
                                    #  X_54_out_index                  # acc : 0.8932926829268293
                                    #  X_55_out_index                  # acc : 0.8932926829268293
                                    #  X_56_out_index                  # acc : 0.8932926829268293
                                     ),axis=None)
print(len(lead_outlier_index)) #577
# print(lead_outlier_index)

lead_not_outlier_index = []
for i in train_x.index:
    if i not in lead_outlier_index :
        lead_not_outlier_index.append(i)
train_x_clean = train_x.loc[lead_not_outlier_index]      
train_x_clean = train_x_clean.reset_index(drop=True)

from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
x_train,x_test,y_train,y_test=train_test_split(train_x,train_y,shuffle=True,random_state=1234,train_size=0.9)

# 2. 모델

n_splits = 5
kfold = KFold(n_splits=n_splits,shuffle=True,random_state=123)

from catboost import CatBoostRegressor
from bayes_opt import BayesianOptimization
# learning_rate : float range: (0,1]
# depth : int, [default=6]   range: [1,+inf]
# od_pval : float, [default=None] range: [0,1]
# model_size_reg : float, [default=None] range: [0,+inf]
# l2_leaf_reg : float, [default=3.0]  range: [0,+inf]
# fold_permutation_block : int, [default=1] T[1, 256]. 
# leaf_estimation_iterations : int, [default=None] range: [1,+inf]
# {'target': 0.09084053546982995, 
#  'params': {'depth': 8.747600921424528, 
#             'fold_permutation_block': 2.426892383721688, 
#             'l2_leaf_reg': 5.197260636886469, 
#             'learning_rate': 0.07742783823042632, 
#             'model_size_reg': 0.4585118086350999, 
#             'od_pval': 0.42664687552652275}}
# lr = MultiOutputRegressor(CatBoostRegressor(random_state=1234,verbose=False))
cat_paramets = {"learning_rate" : [0.07742783823042632],
                'depth' : [8],
                'od_pval' : [0.42664687552652275],
                'fold_permutation_block': [2],
                'model_size_reg': [0.4585118086350999],
                'l2_leaf_reg' :[5.197260636886469]}
from sklearn.multioutput import MultiOutputRegressor
Multi_parameters= {'n_jobs':[-1]}
cat = MultiOutputRegressor(CatBoostRegressor(random_state=123,
                        verbose=False,
                        learning_rate=0.07742783823042632,
                        depth=8,
                        od_pval =0.42664687552652275,
                        fold_permutation_block = 2,
                        model_size_reg = 0.4585118086350999,
                        l2_leaf_reg =5.197260636886469,
                        n_estimators=500))
model = RandomizedSearchCV(cat,Multi_parameters,cv=kfold,n_jobs=-1)

import time 
start_time = time.time()
model.fit(x_train,y_train)   
end_time = time.time()-start_time 
y_predict = model.predict(x_test)
results = r2_score(y_test,y_predict)

print('최적의 매개변수 : ',model.best_params_)
print('최상의 점수 : ',model.best_score_)
print('걸린 시간 : ',end_time)
print('model.socre : ',results)
'''
model.fit(train_x,train_y)
y_summit = model.predict(test_set)
submission = pd.read_csv(path + 'sample_submission.csv',#예측에서 쓸거야!!
                     )

for idx, col in enumerate(submission.columns):
    if col=='ID':
        continue
    submission[col] = y_summit[:,idx-1]
print('Done.')
submission.to_csv('test43.csv',index=False)
'''
# 최적의 매개변수 :  {'learning_rate': 0.1, 'n_estimators': 100}
# 최상의 점수 :  0.06523837244168644
# 걸린 시간 :  60.23209834098816 
# model.socre :  0.0698511473254096   

# 최적의 매개변수 :  {'learning_rate': 0.1, 'n_estimators': 100}
# 최상의 점수 :  0.06577843806588209   
# 걸린 시간 :  309.3102035522461        
# model.socre :  0.0678021185           

# 최적의 매개변수 :  {'learning_rate': 0.1, 'n_estimators': 100}
# 최상의 점수 :  0.06970460808498297    
# 걸린 시간 :  329.5538504123688
# model.socre :  0.07454421428426697