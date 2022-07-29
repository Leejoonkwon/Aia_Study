from keras.preprocessing.text import Tokenizer

<<<<<<< HEAD
text1 = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'


# text5 = ([text3[0],text4[0]]),([text3[1],text4[1]])

token = Tokenizer()

# text에 있는 문자를 읽고 index를 부여해 수치화한다.
(920,2) #1840
print(token.word_index)
# {'마구': 1, '나는': 2, '매우': 3, '또': 4, '진짜': 5, '맛있는'
#  : 6, '밥을': 7, '엄청': 8, '먹었다': 9, '지구용사': 10, '이재근이다': 11, '멋있다': 12, '얘기해봐': 13}
text = [text3,text4]

for i in text:
    text = ([text3[i],text4[i]])
    token.fit_on_texts(text)
    text2 = token.texts_to_sequences(text)
    print(text2)
    
# x = token.texts_to_sequences(text5)
'''
#[[2, 5, 3, 3, 6, 7, 8, 1, 1, 1, 9]]
1,9
##############자연어 처리는 시계열 분석이 기본적이다! 
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
x_new =x[0]+ x[1]
# print(x_new.shape) #(18,)
# x = to_categorical(x_new)
# print(x,x.shape) #(18, 14)

from sklearn.preprocessing import OneHotEncoder
x = np.array(x_new).reshape(-1,1)
ohe = OneHotEncoder()
x = ohe.fit_transform(x).toarray()
# print(x,x.shape) #(18, 13)
# [[2, 5, 3, 3, 6, 7, 8, 1, 1, 1, 9], [2, 10, 11, 12, 4, 4, 13]]
# [[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]] (18, 13)
'''




