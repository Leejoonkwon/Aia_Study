import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from ngboost import NGBRegressor
#1. 데이터
path = './_data/kaggle_shopping/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
print(train_set)

print(train_set.shape) #(6255, 12)

test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)

submission = pd.read_csv(path + 'sample_submission.csv',#예측에서 쓸거야!!
                       index_col=0)

train_set['Date'] = pd.to_datetime(train_set['Date'])
train_set['month'] = train_set['Date'].dt.strftime('%m')
train_set = train_set.drop(['Date'], axis=1)
train_set = train_set.astype({'month':'float64'})
train_set = train_set.fillna(0)
test_set = test_set.fillna(test_set.mean())
target = test_set[['age', 'continent', 'contract_until', 'position', 'reputation', 'stat_overall', 'stat_potential', 'stat_skill_moves']]
kf = KFold(n_splits = 10, random_state = 521, shuffle = True)
cb = CatBoostRegressor(random_state = 521, silent = True, depth = 3)
cb_pred = np.zeros((target.shape[0]))
rmse_list = []
for tr_idx, val_idx in kf.split(X, y) :
    tr_x, tr_y = X.iloc[tr_idx], y.iloc[tr_idx]
    val_x, val_y = X.iloc[val_idx], y.iloc[val_idx]
    
    cb.fit(tr_x, tr_y)
    pred = np.expm1([0 if x < 0 else x for x in cb.predict(val_x)])

    rmse = np.sqrt(mean_squared_error(np.expm1(val_y), pred))
    rmse_list.append(rmse)
    
    sub_pred = np.expm1([0 if x < 0 else x for x in cb.predict(target)]) / 10
    cb_pred += sub_pred
print(f'{cb.__class__.__name__}의 10fold 평균 RMSE는 {np.mean(rmse_list)}')