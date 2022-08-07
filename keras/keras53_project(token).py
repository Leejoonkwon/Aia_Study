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

path = 'D:\study_data/' # ".은 현재 폴더"
df = pd.read_csv(path + 'music2.csv'
                       )
from random import *
# print(df
# print(df.describe) #[400 rows x 4 columns]
# token1 = sent_tokenize()
# is_bal = df['Genre'] == '발라드'

# # 조건를 충족하는 데이터를 필터링하여 새로운 변수에 저장합니다.
# bal = df[is_bal]
# i = randrange(40)  # 0부터 9 사이의 임의의 정수
# print(i)
# bal = '{} - {}'.format(bal['title'][i],bal['artist'][i])
# # 결과를 출력합니다.
# print(bal)
print(df.info()) #400
print(np.unique(df['Genre'],return_counts=True))
#'댄스', '랩,힙합', '록,메탈', '발라드', '알앤비,소울', '인디', '트로트', '포크,블루스'
'''
print("가사의 최대길이 :",max(len(i) for i in df['lyric']))         #2623
print("가사의 평균길이 :",sum(map(len,df['lyric'])) / len(df['lyric']))# 717.3975

token = Tokenizer(oov_token="<OOV>") #oov = out of vocabulary 
df_1 = df['lyric']

token.fit_on_texts(df_1)
x = token.texts_to_sequences(df_1)
# print(token.word_index)
word_size = len(token.word_index)
# print("wored_size :",word_size) #단어 사전의 갯수 : 15237
print(len(x[2])) #[0] 175 #[1] 113 #[2] 182
from keras.preprocessing.sequence import pad_sequences
pad_x =pad_sequences(x,padding='pre',maxlen=170)
print(pad_x.shape) #(400, 170)

le = LabelEncoder()
df['Genre'] = le.fit_transform(df['Genre'])
y = df['Genre']

print(np.unique(pad_x,return_counts=True)) #15238
pad_x = pad_x.reshape(400,170,1)


x_train,x_test,y_train,y_test= train_test_split(pad_x,y,train_size=0.8,shuffle=True,random_state=100)
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape,y_test.shape) #(320, 8) (80, 8)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM,Dense,Embedding
#Embedding이란 원핫 인코딩 필요없이 문자간 상관 관계를 고려해 벡터를 부여한다.그리고 Embedding은 인풋 레이어에서 시작하며 
#총 개수를 input_dim으로 한다.

model = Sequential()
model.add(Embedding(input_dim=15238,output_dim=10,input_length=170)) #단어사전의 갯수 * output_dim(아우풋 노드) =파라미터
model.add(LSTM(32))
model.add(Dense(8,activation='softmax'))
model.summary() #Total params: 5,847

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train,epochs=100,batch_size=100)

#4. 평가, 예측
acc = model.evaluate(x_test,y_test)[1]
print('acc :',acc)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict,axis=1)
y_test = np.argmax(y_test,axis=1)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc 스코어 :', acc)
'''