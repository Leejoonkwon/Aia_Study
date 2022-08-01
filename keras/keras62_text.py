import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import sent_tokenize

xy = pd.read_table("D:/study_data/naver_shopping.txt")

print(xy)# [199999 rows x 2 columns]

x = xy['document']
y = xy['label']
# print(train_data.shape)  #(14000, 2)
# print(train_data['document'].nunique(), train_data['label'].nunique()) # 13999 4
# print(train_data['label'].value_counts().plot(kind = 'bar'))
# print(train_data.groupby('label').size().reset_index(name = 'count'))
# print(train_data.isnull().sum())
print(x)