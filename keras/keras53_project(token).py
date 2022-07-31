from keras.preprocessing.text import Tokenizer
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer


path = 'D:\study_data\_data/' # ".은 현재 폴더"
df = pd.read_csv(path + 'music.csv'
                       )

print(df.info()) 
print(df.describe) #[400 rows x 4 columns]

<<<<<<< HEAD
token = Tokenizer(oov_token="<OOV>") #oov = out of vocabulary 
token.fit_on_texts(df)
x = token.texts_to_sequences(df)
print(x[2][2])
=======
print(data.shape) #(900, 4)
print(data.shape) #(900, 4)

data = data.reshape(1800,)
print(data.shape) #(1800,)
>>>>>>> d8f89b0b8e5a222ae63e2cbaa131e3ca55357985


