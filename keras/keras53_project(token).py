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


path = 'D:\study_data\_data/' # ".은 현재 폴더"
df = pd.read_csv(path + 'music.csv'
                       )

print(df.info()) 
print(df.describe) #[400 rows x 4 columns]

token = Tokenizer(oov_token="<OOV>") #oov = out of vocabulary 
# token.fit_on_texts(df)
# x = token.texts_to_sequences(df)
# print(x)
le = LabelEncoder()
df['Genre'] = le.fit_transform(df['Genre'])
# print(df['Genre'])
df_1 = df['lyric']
df_2 = df['title']
df_3 = df['artist']
df_label = df['Genre']
token.fit_on_texts(df_1)
x = token.texts_to_sequences(df_1)
# print(x)
y = token.texts_to_sequences(df_2)
z = token.texts_to_sequences(df_3)

x = np.array(x)
y = np.array(y)
z = np.array(z)
print(x.shape,y.shape,z.shape) #(400,) (400,) (400,)
data = pd.concat([x,y,z], join='inner', axis=1)
print(data.info())