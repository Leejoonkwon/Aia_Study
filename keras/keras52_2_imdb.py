from keras.datasets import imdb
import numpy as np
import pandas as pd
from tensorflow.python.keras.layers import LSTM,Dense,Embedding

(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=10000)

print(x_train,x_train.shape,x_test.shape) #(25000,) (25000,)
print(y_train,y_train.shape) #(25000,) 
print(len(np.array(x_train))) #25000
print(len(np.unique(y_train))) #2개
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape)

print("뉴스기사의 최대길이 :",max(len(i) for i in x_train))         #뉴스기사의 최대길이 : 2494
print("뉴스기사의 평균길이 :",sum(map(len,x_train)) / len(x_train)) #뉴스기사의 평균길이 : 238.71364

#2. 모델 구성
from tensorflow.python.keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

#[맹그러오]
x_train = pad_sequences(x_train,padding='pre',maxlen=100,truncating='pre')
x_test = pad_sequences(x_test,padding='pre',maxlen=100,truncating='pre')
model = Sequential()
model.add(Embedding(input_dim=46,output_dim=10,input_length=100)) #단어사전의 갯수 * output_dim(아웃풋 노드) =파라미터
# input_dim이 꼭 단어 갯수와 일치해야하는 것은 아니지만 가급적 맞춰야 좋다.
# model.add(Embedding(input_dim=31,output_dim=10)) #length를 명시하지 않아도 N개로 인식해서 실행한다.
# model.add(Embedding(31,10)) # 명시하지 않아도 위치에 따라 옵션을 자동으로 인식한다.
# model.add(Embedding(31,10,5)) # error input_length는 명시해야 한다.
# model.add(Embedding(31,3,input_length = 5)) 
model.add(LSTM(32))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
<<<<<<< HEAD
model.add(Dense(1,activation='sigmoid'))
=======
model.add(Dense(2,activation='sigmoid'))
>>>>>>> bc514901d26e19fad45765b4f2686f98230492c7
model.summary() #Total params: 5,847

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train,epochs=3,batch_size=5000)

#4. 평가, 예측
acc = model.evaluate(x_test,y_test)[1]
print('acc :',acc)
# y_predict = model.predict(x_test)
# print('predict :',y_predict)
# acc : 0.5599600076675415
#
# acc : 0.5416799783706665

