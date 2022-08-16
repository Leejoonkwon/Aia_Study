# 만났던 문제점들 
"""
ValueError: Data cardinality is ambiguous:
  x sizes: 28, 28, 28, 28, 24
  y sizes: 28, 28, 28, 28, 24
Make sure all arrays contain the same number of samples.
reshape 입력 형태불일치로 인한 에러  https://github.com/tensorflow/tensorflow/issues/38702



"""
### 영화 매출 예측 앙상블 모델 ###

#1_1.[ 데이터 불러오기 ]###################################################################
import pandas as pd
탑건 = pd.read_csv('C:\\review_data\영화 훈련 데이터/탑건.csv', sep='\t') 
아이언맨 = pd.read_csv('C:\\review_data\영화 훈련 데이터/아이언맨.csv', sep='\t') 
어벤져스 = pd.read_csv('C:\\review_data\영화 훈련 데이터/어벤져스.csv', sep='\t') 
엔드게임 = pd.read_csv('C:\\review_data\영화 훈련 데이터/엔드게임.csv', sep='\t') 
용의출현 = pd.read_csv('C:\\review_data\영화 훈련 데이터/토르.csv', sep='\t') 

# print(탑건.shape)       # (45, 7)
# print(아이언맨.shape)   # (118, 7)
# print(어벤져스.shape)   # (134, 7)
# print(엔드게임.shape)   # (126, 7)
# print(용의출현.shape)   # (31, 7)

#1_2.[ 날짜 데이터 분리 ]###################################################################

탑건['일자'] = pd.to_datetime(탑건['날짜'])
탑건['연도'] = 탑건['일자'].dt.year
탑건['월'] = 탑건['일자'].dt.month
탑건['일'] = 탑건['일자'].dt.day

아이언맨['일자'] = pd.to_datetime(아이언맨['날짜'])
아이언맨['연도'] = 아이언맨['일자'].dt.year
아이언맨['월'] = 아이언맨['일자'].dt.month
아이언맨['일'] = 아이언맨['일자'].dt.day

어벤져스['일자'] = pd.to_datetime(어벤져스['날짜'])
어벤져스['연도'] = 어벤져스['일자'].dt.year
어벤져스['월'] = 어벤져스['일자'].dt.month
어벤져스['일'] = 어벤져스['일자'].dt.day

엔드게임['일자'] = pd.to_datetime(엔드게임['날짜'])
엔드게임['연도'] = 엔드게임['일자'].dt.year
엔드게임['월'] = 엔드게임['일자'].dt.month
엔드게임['일'] = 엔드게임['일자'].dt.day

용의출현['일자'] = pd.to_datetime(용의출현['날짜'])
용의출현['연도'] = 용의출현['일자'].dt.year
용의출현['월'] = 용의출현['일자'].dt.month
용의출현['일'] = 용의출현['일자'].dt.day


#1_3.[ 데이터 안에 0인 값 확인 ]###################################################################

for col in 탑건.columns:
    missing_rows = 탑건.loc[탑건[col]==0].shape[0]
    # print(col + ': ' + str(missing_rows))
for col in 아이언맨.columns:
    missing_rows = 아이언맨.loc[아이언맨[col]==0].shape[0]
    # print(col + ': ' + str(missing_rows))
for col in 어벤져스.columns:
    missing_rows = 어벤져스.loc[어벤져스[col]==0].shape[0]
    # print(col + ': ' + str(missing_rows))
for col in 엔드게임.columns:
    missing_rows = 엔드게임.loc[엔드게임[col]==0].shape[0]
    # print(col + ': ' + str(missing_rows))
for col in 용의출현.columns:
    missing_rows = 용의출현.loc[용의출현[col]==0].shape[0]
    # print(col + ': ' + str(missing_rows))
# 0인 값은 없는 것으로 판명


#1_4.[ nan 확인 및 제거 ]###################################################################
# print(탑건.isnull().sum())  
# print(용의출현.isnull().sum())  

# 없는 것으로 판명



#1_5.[ y데이터 분리 ]###################################################################

탑건_매출액 = 탑건['매출액']
y1 = 아이언맨['매출액']
y2 = 어벤져스['매출액']
y3 = 엔드게임['매출액']
용의출현_매출액 = 용의출현['매출액']

#1_6.[ x데이터 컬럼 정리 및 shape 확인 ]###################################################################

# 탑건 = 탑건.drop(['일자', '매출액', '날짜'], axis=1)
x1 = 아이언맨.drop(['일자', '매출액', '날짜'], axis=1)
x2 = 어벤져스.drop(['일자', '매출액', '날짜'], axis=1)
x3 = 엔드게임.drop(['일자', '매출액', '날짜'], axis=1)
# 용의출현 = 용의출현.drop(['일자', '매출액', '날짜'], axis=1)

# print(탑건.shape)       # (45, 8)
# print(아이언맨.shape)   # (31, 8)
# print(어벤져스.shape)   # (31, 8)
# print(엔드게임.shape)   # (45, 8)
# print(용의출현.shape)   # (31, 8)


#1_7.[ 데이터 정규화 ]###################################################################



# 정규화 대상 column 정의
# scale_cols = ['상영횟수', '좌석수', '관객수', '누적매출액', '누적관객수', '연도', '월', '일']

# 탑건_scaler = scaler.fit_transform(탑건[scale_cols])
# 아이언맨_scaler = scaler.fit_transform(아이언맨[scale_cols])
# 어벤져스_scaler = scaler.fit_transform(어벤져스[scale_cols])
# 엔드게임_scaler = scaler.fit_transform(엔드게임[scale_cols])
# 용의출현_scaler = scaler.fit_transform(용의출현[scale_cols])

# print(탑건_scaler)
# print(용의출현_scaler)


#1_8[DataFrame을 numpy로 변환 작업 ]#############################################################

# 탑건_scaler = pd.DataFrame(탑건_scaler, columns=scale_cols)
# 탑건_scaler = 탑건_scaler.to_numpy()
# 탑건_매출액 = 탑건_매출액.to_numpy()

# 아이언맨_scaler = pd.DataFrame(아이언맨_scaler, columns=scale_cols)
# 아이언맨_scaler = 아이언맨_scaler.to_numpy()
# 아이언맨_매출액 = 아이언맨_매출액.to_numpy()

# 어벤져스_scaler = pd.DataFrame(어벤져스_scaler, columns=scale_cols)
# 어벤져스_scaler = 어벤져스_scaler.to_numpy()
# 어벤져스_매출액 = 어벤져스_매출액.to_numpy()

# 엔드게임_scaler = pd.DataFrame(엔드게임_scaler, columns=scale_cols)
# 엔드게임_scaler = 엔드게임_scaler.to_numpy()
# 엔드게임_매출액 = 엔드게임_매출액.to_numpy()

# 용의출현_scaler = pd.DataFrame(용의출현_scaler, columns=scale_cols)
# 용의출현_scaler = 용의출현_scaler.to_numpy()
# 용의출현_매출액 = 용의출현_매출액.to_numpy()
import numpy as np

def generator(data, window, offset):
    gen = data.to_numpy() #데이터 프레임을 배열객체로 반환
    X = []
    
    for i in range(len(gen)-window-offset): # 420522
        row = [[a] for a in gen[i:i+window]] #행
        X.append(row)
        
        
    return np.array(X)


WINDOW = 6
OFFSET = 20
print(x1.shape,x1) #(118, 8)
#아이언맨
aaa1 = generator(x1,WINDOW,OFFSET) #x1를 위한 데이터 drop 제외 모든 amo에 데이터
x_1 = aaa1[:,:-1]
bbb = generator(y1,WINDOW,OFFSET)
y_1 = bbb[:,-1]
# 어벤져스
aaa2 = generator(x2,WINDOW,OFFSET) #x1를 위한 데이터 drop 제외 모든 amo에 데이터
x_2 = aaa2[:,:-1]
bbb = generator(y2,WINDOW,OFFSET)
y_2 = bbb[:,-1]
# 엔드게임
aaa3 = generator(x3,WINDOW,OFFSET) #x1를 위한 데이터 drop 제외 모든 amo에 데이터
x_3 = aaa3[:,:-1]
bbb = generator(y3,WINDOW,OFFSET)
y_3 = bbb[:,-1]




#1_9[ train, test 분리 ]#############################################################################
from sklearn.model_selection import train_test_split
x1_train,x1_test,x2_train,x2_test,x3_train,x3_test,y1_train,y1_test,y2_train,y2_test,y3_train,y3_test \
=train_test_split(x_1,x_2,x_3,y_1,y_2,y_3,shuffle=False,train_size=0.85)
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

scaler = MinMaxScaler()


#1_10[ 3차원으로 변환 ]####################################################################################################################3
# x1_train = x1_train.reshape(78,40)
# x2_train = x2_train.reshape(78,40)
# x3_train = x3_train.reshape(78,40)

# x1_test = x1_test.reshape(14,40)
# x2_test = x2_test.reshape(14,40)
# x3_test = x3_test.reshape(14,40)
# x1_train = scaler.fit_transform(x1_train)
# x2_train = scaler.fit_transform(x2_train)
# x3_train = scaler.fit_transform(x3_train)
# x1_test = scaler.transform(x1_test)
# x2_test = scaler.transform(x2_test)
# x3_test = scaler.transform(x3_test)
# print(x1_train.shape,y1_train.shape) #(66, 4, 1, 8) (66, 1)
# print(x2_train.shape,y2_train.shape) #(66, 4, 1, 8) (66, 1)
# print(x3_train.shape,y3_train.shape) #(66, 4, 1, 8) (66, 1)

# print(x1_test.shape,y1_test.shape) #(23, 4, 1, 8) (23, 1)
# print(x2_test.shape,y2_test.shape) #(23, 4, 1, 8) (23, 1)
# print(x3_test.shape,y3_test.shape) #(23, 4, 1, 8) (23, 1)

x1_train = x1_train.reshape(78,5,8)
x2_train = x2_train.reshape(78,5,8)
x3_train = x3_train.reshape(78,5,8)

x1_test = x1_test.reshape(14,5,8)
x2_test = x2_test.reshape(14,5,8)
x3_test = x3_test.reshape(14,5,8)

y1_train = y1_train.reshape(78,)
y2_train = y2_train.reshape(78,)
y3_train = y3_train.reshape(78,)

y1_test = y1_test.reshape(14,)
y2_test = y2_test.reshape(14,)
y3_test = y3_test.reshape(14,)


#2.[ 모델구성 ]###########################################################################################
from tensorflow.python.keras.layers import Dense, LSTM,GRU,Dropout,SimpleRNN,Conv1D
from tensorflow.python.keras.models import Input, Model

# # 2-1. 모델 : 탑건
# 탑건_input = Input(shape=(4,8))    # print(x1_train.shape, x1_test.shape)   
# dense1 = LSTM(128, activation='relu', name='jun1')(탑건_input)
# dense2 = Dense(128, activation='relu', name='jun2')(dense1)
# dense3= Dense(64, activation='relu', name='jun3')(dense2)
# dense4= Dense(128, activation='relu', name='jun4')(dense3)
# 탑건_mid = Dense(64, activation='relu', name='out_jun1')(dense4)

#2-2 모델 아이언맨
아이언맨_input = Input(shape=(5,8))     # print(x2_train.shape, x2_test.shape)
dense11 = Conv1D(64,1,activation='relu')(아이언맨_input)
dense12 = LSTM(128, activation='relu', name='jun11')(dense11)
dense22 = Dense(128, activation='relu', name='jun12')(dense12)
dense32= Dense(64, activation='relu', name='jun13')(dense22)
dense42= Dense(128, activation='relu', name='jun14')(dense32)
아이언맨_mid = Dense(64, activation='relu', name='out_jun2')(dense42)

#2-3 모델 어벤져스
어벤져스_input = Input(shape=(5,8))     # print(x2_train.shape, x2_test.shape)  
dense01 = Conv1D(64,1,activation='relu')(어벤져스_input)
dense13 = LSTM(128, activation='relu', name='jun111')(dense01)
dense23 = Dense(128, activation='relu', name='jun112')(dense13)
dense33= Dense(64, activation='relu', name='jun113')(dense23)
dense43= Dense(128, activation='relu', name='jun114')(dense33)
어벤져스_mid = Dense(64, activation='relu', name='out_jun3')(dense43)

#2-4 모델 엔드게임
엔드게임_input = Input(shape=(5,8))     # print(x2_train.shape, x2_test.shape) 
dense51 = Conv1D(64,1,activation='relu')(엔드게임_input)
dense14 = LSTM(128, activation='relu', name='jun1111')(dense51)
dense24 = Dense(128, activation='relu', name='jun1112')(dense14)
dense34= Dense(64, activation='relu', name='jun1113')(dense24)
dense44= Dense(128, activation='relu', name='jun1114')(dense34)
엔드게임_mid = Dense(64, activation='relu', name='out_jun4')(dense44)

# #2-5 모델 용의출현
# 용의출현_input = Input(shape=(8, 1))     # print(x2_train.shape, x2_test.shape)  
# dense15 = LSTM(128, activation='relu', name='jun11111')(용의출현_input)
# dense25 = Dense(128, activation='relu', name='jun11112')(dense15)
# dense35= Dense(64, activation='relu', name='jun11113')(dense25)
# dense45= Dense(128, activation='relu', name='jun11114')(dense35)
# 용의출현_mid = Dense(64, activation='relu', name='out_jun5')(dense45)


from tensorflow.python.keras.layers import concatenate, Concatenate 

merge1 = concatenate([아이언맨_mid, 어벤져스_mid, 엔드게임_mid], name='mg1')
merge2 = Dense(200, activation='relu', name='mg2_15')(merge1)
merge3 = Dense(300, name='mg3_12')(merge2)
concatenate_output = Dense(10, name='last')(merge3)

output12 =  Dense(100)(concatenate_output)
output22 = Dense(100)(output12)
output201 = Dropout(0.2)(output22)
아이언맨_output = Dense(1, name='last2')(output201)

output13 =  Dense(100)(concatenate_output)
output101 = Dropout(0.2)(output13)
output23 = Dense(100)(output101)
어벤져스_output = Dense(1, name='last3')(output23)

output14 =  Dense(100)(concatenate_output)
output24 = Dense(100)(output14)
output301 = Dropout(0.2)(output24)

엔드게임_output = Dense(1, name='last4')(output301)

from tensorflow.python.keras.models import Model
model = Model(inputs=[아이언맨_input, 어벤져스_input, 엔드게임_input],
              outputs=[아이언맨_output, 어벤져스_output, 엔드게임_output])
# model.summary()



#3.[ 컴파일, 훈련 ]###################################################################################################
model.compile(loss='mae', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# import datetime
# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M") # 0707_1723
# print(date)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# earlyStopping = EarlyStopping(monitor='val_loss', patience=200, mode='auto', verbose=1, 
#                               restore_best_weights=True)        

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
#                       filepath= "".join([filepath, 'k24_', date, '_', filename])
#                       )

hist = model.fit([x1_train, x2_train, x3_train],
                 [y1_train, y2_train, y3_train],
                 epochs=150,
                 batch_size=30,
                 validation_split=0.25,
                 verbose=2)
# model.save('./_ModelCheckPoint/keras24_ModelCheckPoint.hdf5')

loss = model.evaluate([x1_test,x2_test,x3_test], [y1_test,y2_test,y3_test])
print(loss)

y_predict = model.predict([x1_test,x2_test,x3_test])
# print(y_predict)
from sklearn.metrics import r2_score

r1 = r2_score(y1_test,y_predict[0])
r2 = r2_score(y2_test,y_predict[1])
r3 = r2_score(y3_test,y_predict[2])

print(y_predict[0][-1])
print('r2스코어 :', r1)
print('r2스코어 :', r2)
print('r2스코어 :', r3)
