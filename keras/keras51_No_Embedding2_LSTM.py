from keras.preprocessing.text import Tokenizer
import numpy as np          

#1. 데이터
docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화에요'
        , '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글세요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', ' 너무 재미없다.','참 재밌네요', '민수가 못 생기기긴 했어요'
        ,'안결 혼해요']

# 긍정 1,부정 0
label= np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0]) # (14,)
 
token = Tokenizer()
token.fit_on_texts(docs)
# print(token.word_index)
# {'참': 1, '너무': 2, '재밌어요': 3, '최고에요': 4, '잘': 5, '만든': 6, '영화에요': 7, 
#  '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, '
#  싶네요': 15, '글세요': 16, '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, 
#  '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24, '민수가': 25, '못': 26,
#  '생기기긴': 27, '했어요': 28, '안결': 29, '혼해요': 30}

x = token.texts_to_sequences(docs)
print(x) 
#각 문장마다 길이가 다르기 때문에 가장 큰 값을 기준으로 모자른 값들은 0으로 채운다.
#값이 너무 클경우 일정양으로 자른다.
# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15],
#  [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 26, 27, 28], [29, 30]]
from keras.preprocessing.sequence import pad_sequences
pad_x =pad_sequences(x,padding='pre',maxlen=5) #통상 같은 크기를 맞추기 위해서는 앞에서부터 패딩한다. maxlen 최대 글자 수의 제한 
print(pad_x)
print(pad_x.shape) #(14, 5)
word_size = len(token.word_index)
print("wored_size :",word_size) #단어 사전의 갯수 : 30

print(np.unique(pad_x,return_counts=True))
# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]), 
# array([37,  3,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
#         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],

#2. 모델 구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM,Dense,Embedding
#Embedding이란 원핫 인코딩 필요없이 문자간 상관 관계를 고려해 벡터를 부여한다.그리고 Embedding은 인풋 레이어에서 시작하며 
#총 개수를 input_dim으로 한다.
pad_x = pad_x.reshape(14,5,1) 
#Embedding을 쓰지 않는 경우 중 RNN을 모델로 사용한다면 3차원 데이터로 가공이 필요하다.성능은 Embedding가 더 좋다.
model = Sequential()
# model.add(Embedding(input_dim=31,output_dim=10,input_length=5)) #단어사전의 갯수 * output_dim(아우풋 노드) =파라미터
# input_dim이 꼭 단어 갯수와 일치해야하는 것은 아니지만 가급적 맞춰야 좋다.
# model.add(Embedding(input_dim=31,output_dim=10)) #length를 명시하지 않아도 N개로 인식해서 실행한다.
# model.add(Embedding(31,10)) # 명시하지 않아도 위치에 따라 옵션을 자동으로 인식한다.
# model.add(Embedding(31,10,5)) # error input_length는 명시해야 한다.
# model.add(Embedding(31,3,input_length = 5)) 
model.add(LSTM(32,input_shape=(5,1)))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary() #Total params: 5,847

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(pad_x,label,epochs=20,batch_size=16)

#4. 평가, 예측
acc = model.evaluate(pad_x,label)[1]
print('acc :',acc)


# acc : 0.8571428656578064