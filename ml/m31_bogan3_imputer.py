import pandas as pd 
import numpy as np 
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

data = pd.DataFrame([[2,np.nan,6,8,10],
                     [2, 4,np.nan,8,np.nan],
                     [2,4,6,8,10],
                     [np.nan,4,np.nan,8,np.nan,],
                     ])
data = data.transpose()
data.columns = ['x1','x2','x3','x4']
# imputer = SimpleImputer() # 디폴트는 평균값으로 대치
# imputer = SimpleImputer(strategy='mean') # 디폴트는 평균값으로 대치
# imputer = SimpleImputer(strategy='median') # 중위값으로 대치
# imputer = SimpleImputer(strategy='constant') # 상수 0으로 대치
imputer = KNNImputer() # 
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN
# [[ 2.          2.          2.          6.        ]
#  [ 6.5         4.          4.          4.        ]
#  [ 6.          4.66666667  6.          6.        ]
#  [ 8.          8.          8.          8.        ]
#  [10.          4.66666667 10.          6.        ]]
# imputer = IterativeImputer() 
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN
# [[ 2.          2.          2.          2.0000005 ]
#  [ 4.00000099  4.          4.          4.        ]
#  [ 6.          5.99999928  6.          5.9999996 ]
#  [ 8.          8.          8.          8.        ]
#  [10.          9.99999872 10.          9.99999874]]
imputer.fit(data)
print(data)

data3 = imputer.transform(data)
print(data3)
# data = data.transpose()
# data.columns = ['x1','x2','x3','x4']
# print(data.shape)
# print(data)
