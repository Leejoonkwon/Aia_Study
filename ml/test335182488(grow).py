import pandas as pd
import numpy as np
import glob
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM,GRU,Dense,Dropout
from sklearn.model_selection import train_test_split

path = 'D:\study_data\_data\_csv\dacon_grow/'
all_input_list = sorted(glob.glob(path + 'train_input/*.csv'))
all_target_list = sorted(glob.glob(path + 'train_target/*.csv'))

import pandas as pd
from glob import glob

# file_names = glob("data/*.csv") #폴더 내의 모든 csv파일 목록을 불러온다
total = pd.DataFrame() #빈 데이터프레임 하나를 생성한다
total2 = pd.DataFrame() #빈 데이터프레임 하나를 생성한다

for file_name in all_input_list:
    temp = pd.read_csv(file_name, encoding='utf-8',index_col=0) #csv파일을 하나씩 열어 임시 데이터프레임으로 생성한다
    total = pd.concat([total, temp]) #전체 데이터프레임에 추가하여 넣는다
    
    
    
for file_name in all_target_list:
    temp = pd.read_csv(file_name, encoding='utf-8',index_col=0) #csv파일을 하나씩 열어 임시 데이터프레임으로 생성한다
    total2 = pd.concat([total2, temp]) #전체 데이터프레임에 추가하여 넣는다
    
# df_summary = pd.DataFrame()

total = pd.to_datetime(total.index)
print(total)

# df_resample = total.resample(rule='H').last()
# print(df_resample)








