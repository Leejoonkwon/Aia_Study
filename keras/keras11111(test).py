import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
#1. 데이터
path = 'D:\study_data\_data/' # ".은 현재 폴더"
df = pd.read_csv(path + 'music.csv',
                        )

print(df.shape) #(400, 3)

print(df.info())

token = Tokenizer(oov_token="<OOV>") #oov = out of vocabulary 
print(df.shape)

token.fit_on_texts(df['lyric'])
x1 = token.texts_to_sequences(df['lyric'])
x2 = token.texts_to_sequences(df['title'])
x3 = token.texts_to_sequences(df['artist'])
print(x2)
y = df['class']
print(y)
# token.fit_on_texts(data) # 나는 코딩 고수다.  1:나는 2:코딩  3:고수다
# print(data[:1])
# print(token.word_index)
# data = data.reshape(900,2)
# print(data.shape) #900,2
# # data = np.array(data)
# print(data[0][1])

# x = data
# y = token.texts_to_sequences(x)# ( 1 2 3) # ( 5 2 1 )
# print(y)

