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

# print(xy)# [199999 rows x 2 columns]

x = xy['document']
y = xy['label']

# print(x)
token = Tokenizer(oov_token="<OOV>") #oov = out of vocabulary 
# df_1 = df['lyric']

token.fit_on_texts(x)
x = token.texts_to_sequences(x)
word_size = len(token.word_index)
# print(token.word_index) #
x_train,x_test,y_trian,y_test = train_test_split(x,y,train_size=0.8,shuffle=True,random_state=100)
print("wored_size :",word_size) # 단어 사전의 갯수 : 364926
print("뉴스기사의 최대길이 :",max(len(i) for i in x_train))         #뉴스기사의 최대길이 : 47
print("뉴스기사의 평균길이 :",sum(map(len,x_train)) / len(x_train)) #뉴스기사의 평균길이 : 8.678922993268708
'''
print(len(x[2])) #[0] 175 #[1] 113 #[2] 182
from keras.preprocessing.sequence import pad_sequences
pad_x =pad_sequences(x,padding='pre',maxlen=170)
print(pad_x.shape) #(400, 170)

le = LabelEncoder()
df['Genre'] = le.fit_transform(df['Genre'])
y = df['Genre']

print(np.unique(pad_x,return_counts=True)) #15238
pad_x = pad_x.reshape(400,170,1)


'''
