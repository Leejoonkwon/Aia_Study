from keras.preprocessing.text import Tokenizer
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
path = 'C:\Study\Study\keras/' # ".은 현재 폴더"
df = pd.read_csv(path + 'music.csv',
                        index_col=0)
token = Tokenizer(oov_token="<OOV>") #oov = out of vocabulary 
token.fit_on_texts(df)
x = token.texts_to_sequences(df)
print(x)
