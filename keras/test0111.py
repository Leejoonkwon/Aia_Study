from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Reshape,LSTM,Conv1D,Input
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
import pandas as pd
import tensorflow as tf

#1. 데이터
path = './_data/kaggle_jena/' # ".은 현재 폴더"
df = pd.read_csv(path + 'jena_climate_2009_2016.csv' )

def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []
    print('=== start_index:', start_index)
    print('=== end_index:', end_index)
    print('=== history_size:', history_size)
    print('=== target_size:', target_size)
    print('=== len(dataset):', len(dataset))
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    print('=== end_index2:', end_index)
    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        # np.reshape : 행렬의 형식 변환  history_size 행에서 => history_size 행 1열
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    #print('=== labels:', labels)
    return np.array(data), np.array(labels)

TRAIN_SPLIT = 300000
tf.random.set_seed(13)

