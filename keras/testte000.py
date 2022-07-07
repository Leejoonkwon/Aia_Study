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
                        index_col= 0)
test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                        index_col= 0)

plt.figure(figsize=(10, 10))
sns.heatmap(train_set.corr(), linewidths=0.01, square=True,
            annot=True, cmap=plt.cm.viridis, linecolor="white")
#숫자 표기 여부
plt.title('Correlation between features')
plt.show()




