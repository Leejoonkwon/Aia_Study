from keras.preprocessing.text import Tokenizer
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
path = 'C:\Study\Study\keras/' # ".은 현재 폴더"
df = pd.read_csv(path + 'music.csv'
                       )
df.iloc
data = df.drop(['순위','artist'],axis=1)

data = np.array(data)
# 가수,장르,타이틀,가사
print(data.shape) #(900, 2)
print(data)

data = data.reshape(1800,)
print(data.shape) #(1800,)

