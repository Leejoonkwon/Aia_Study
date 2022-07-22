from keras.datasets import reuters
import numpy as np
import pandas as pd
from tensorflow.python.keras.layers import LSTM,Dense,Embedding

(x_train,y_train),(x_test,y_test) = reuters.load_data(num_words=10000,test_split=0.2)
#이것은 46개 이상의 주제로 분류된 Reuters의 11,228개 뉴스와이어 데이터 세트입니다.

# print(x_train,x_train.shape,x_test.shape) #(8982,) (2246,)
# print(y_train,y_train.shape) #(8982,) 
# print(len(np.array(x_train))) #8982
# print(len(np.unique(y_train))) #46개

# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
#        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45], dtype=int64),
# array([  55,  432,   74, 3159, 1949,   17,   48,   16,  139,  101,  124,
#         390,   49,  172,   26,   20,  444,   39,   66,  549,  269,  100,
#          15,   41,   62,   92,   24,   15,   48,   19,   45,   39,   32,
#          11,   50,   10,   49,   19,   19,   24,   36,   30,   13,   21,
#          12,   18], dtype=int64))
# print(type(x_train),type(y_train)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>
# print(type(x_train[0]),type(y_train[0])) #<class 'list'> <class 'numpy.int64'> #list가 일정하지 않을 때
# print(x_train[0].shape) #AttributeError: 'list' object has no attribute 'shape'
# print(len(x_train[0])) #87
# print(len(x_train[1])) #56
# print(len(x_train[2])) #139
print("뉴스기사의 최대길이 :",max(len(i) for i in x_train))         #뉴스기사의 최대길이 : 2376
print("뉴스기사의 평균길이 :",sum(map(len,x_train)) / len(x_train)) #뉴스기사의 평균길이 : 145.5398574927633
#전처리
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train,padding='pre',maxlen=100,truncating='pre')
x_test = pad_sequences(x_test,padding='pre',maxlen=100,truncating='pre')
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape,x_test.shape) #(8982, 100) (8982, 100)
print(y_train.shape,y_test.shape) #(8982, 46) (2246, 46)

#2. 모델 구성
from tensorflow.python.keras.models import Sequential
#[맹그러오]
model = Sequential()
model.add(Embedding(input_dim=46,output_dim=10,input_length=100)) #단어사전의 갯수 * output_dim(아우풋 노드) =파라미터
# input_dim이 꼭 단어 갯수와 일치해야하는 것은 아니지만 가급적 맞춰야 좋다.
# model.add(Embedding(input_dim=31,output_dim=10)) #length를 명시하지 않아도 N개로 인식해서 실행한다.
# model.add(Embedding(31,10)) # 명시하지 않아도 위치에 따라 옵션을 자동으로 인식한다.
# model.add(Embedding(31,10,5)) # error input_length는 명시해야 한다.
# model.add(Embedding(31,3,input_length = 5)) 
model.add(LSTM(32))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(46,activation='softmax'))
model.summary() #Total params: 5,847

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train,epochs=500,batch_size=2500)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss :',loss)
y_predict = model.predict(x_test)
print('predict :',np.round(y_predict[-1],0))
