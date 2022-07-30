from keras.preprocessing.text import Tokenizer

text = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'

token = Tokenizer()

token.fit_on_texts([text]) # text에 있는 문자를 읽고 index를 부여해 수치화한다.

print(token.word_index)
#{'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}
x = token.texts_to_sequences([text])
print(x)

#[[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]] (10,1)
##############자연어 처리는 시계열 분석이 기본적이다!
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# x = to_categorical(x)
# print(x,x.shape) #(1, 11, 9)
from sklearn.preprocessing import OneHotEncoder
import numpy as np
x = np.array(x).reshape(-1,1)
ohe = OneHotEncoder()
x = ohe.fit_transform(x).toarray()
print(x,x.shape) #(11, 8)
# [[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]]
# [[0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1.]] (11, 8)





