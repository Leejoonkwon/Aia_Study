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

print(data.shape) #(900, 2)
print(data)

data = data.reshape(1800,)
print(data.shape) #(1800,)


# data = df.drop(['장르'],axis=1)
# print(data.shape)# (900,2)

# label = df['장르']
# print(label.shape)# (900,)
# x = data
# y = label
# x_train,x_test,y_train

token = Tokenizer(oov_token="<OOV>") #oov = out of vocabulary 
<<<<<<< HEAD
print(df.shape)
'''
token.fit_on_texts(df)
x = token.texts_to_sequences(df)
print(x)
'''
=======
token.fit_on_texts(data) # 나는 코딩 고수다.  1:나는 2:코딩  3:고수다
print(data[:1])
print(token.word_index)
data = data.reshape(900,2)
print(data.shape) #900,2
# data = np.array(data)
print(data[0][1])
'''
x = data
y = token.texts_to_sequences(x)# ( 1 2 3) # ( 5 2 1 )
print(y)
'''
>>>>>>> c3759647974a215fc0589295540965ae5534e537
