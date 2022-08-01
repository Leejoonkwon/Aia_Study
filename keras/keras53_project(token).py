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

path = 'D:\study_data\_data/' # ".은 현재 폴더"
df = pd.read_csv(path + 'music.csv'
                       )

print(df.info()) 
print(df.describe) #[400 rows x 4 columns]
# token1 = sent_tokenize()
token = Tokenizer(oov_token="<OOV>") #oov = out of vocabulary 
df_1 = df['lyric']

# token.fit_on_texts(df_1)
# x = token.texts_to_sequences(df_1)
print(sent_tokenize(df_1))
''''
le = LabelEncoder()
df['Genre'] = le.fit_transform(df['Genre'])
# print(df['Genre'])
df_1 = df['lyric']
df_2 = df['title']
df_3 = df['artist']
y_data = df['Genre']
token.fit_on_texts(df_1)
x = token.texts_to_sequences(df_1)
print(x)
print(token.word_index)
word_size = len(token.word_index)
print("wored_size :",word_size) #단어 사전의 갯수 : 15237
from keras.preprocessing.sequence import pad_sequences
pad_x =pad_sequences(x,padding='pre',maxlen=5)
print(np.unique(pad_x,return_counts=True))
print(pad_x.shape)


y = token.texts_to_sequences(df_2)
z = token.texts_to_sequences(df_3)
x= np.array(x)
y= np.array(y)
z= np.array(z)
print(x.shape,y.shape,z.shape) #(400,) (400,) (400,)
data = list(map(list.__add__,x,y))
x_data = list(map(list.__add__,data,z))
x_data = np.array(x_data)


x_train,x_test,y_train,y_test= train_test_split(x_data,y_data,train_size=0.8,shuffle=True,random_state=100)
print(x_train)
'''