#### evalu까지만
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Reshape,LSTM,Conv1D,Input
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from sklearn.preprocessing import LabelEncoder
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
import pandas as pd

#1. 데이터
path = './_data/kaggle_jena/' # ".은 현재 폴더"
df = pd.read_csv(path + 'jena_climate_2009_2016.csv' )


# print(df.info())
#  #   Column           Non-Null Count   Dtype
# ---  ------           --------------   -----
#  0   p (mbar)         420551 non-null  float64
#  1   T (degC)         420551 non-null  float64
#  2   Tpot (K)         420551 non-null  float64
#  3   Tdew (degC)      420551 non-null  float64
#  4   rh (%)           420551 non-null  float64
#  5   VPmax (mbar)     420551 non-null  float64
#  6   VPact (mbar)     420551 non-null  float64
#  7   VPdef (mbar)     420551 non-null  float64
#  8   sh (g/kg)        420551 non-null  float64
#  9   H2OC (mmol/mol)  420551 non-null  float64
#  10  rho (g/m**3)     420551 non-null  float64
#  11  wv (m/s)         420551 non-null  float64
#  12  max. wv (m/s)    420551 non-null  float64
#  13  wd (deg)         420551 non-null  float64
# print(df.describe) #[420551 rows x 14 columns]>

df['Date Time'] = pd.to_datetime(df['Date Time'])

df['year'] = df['Date Time'].dt.strftime('%Y')
df['month'] = df['Date Time'].dt.strftime('%m')
df['day'] = df['Date Time'].dt.strftime('%d')
df['hour'] = df['Date Time'].dt.strftime('%h')
df['minute'] = df['Date Time'].dt.strftime('%M')

print(df)
df = df.drop(['Date Time'],axis=1)

print(df.shape)
print('=================')

cols = ['year','month','day','hour','minute']
for col in cols:
    le = LabelEncoder()
    df[col]=le.fit_transform(df[col])
    
size = 10
def split_x(dataset, size): # def라는 예약어로 split_x라는 변수명을 아래에 종속된 기능들을 수행할 수 있도록 정의한다.
    aaa = []
    #aaa 는 []라는 값이 없는 리스트임을 정의
    for i in range(len(dataset)- size + 1): # 6이다 range(횟수)
        #for문을 사용하여 반복한다.첫문장에서 정의한 dataset을 
        subset = dataset[i : (i + size)]
        #i는 처음 0에 개념 [0:0+size]
        # 0~(0+size-1인수 까지 )노출 
        aaa.append(subset) #append 마지막에 요소를 추가한다는 뜻
        #aaa는  []의 빈 리스트 이니 subset이 aaa의 []안에 들어가는 것
        #aaa 가 [1,2,3]이라면  aaa.append(subset)은 [1,2,3,subset]이 될 것이다.
    return np.array(aaa)    

bbb = split_x(df, size)

x = bbb[:, :-1]

y = bbb[:, -1]


print(x,x.shape) #(420542, 9, 19)
print(y,y.shape) #(420542, 19)
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                     train_size=0.75,random_state=100)
print(x_train.shape,x_test.shape) #(315406, 9, 19) (105136, 9, 19)

'''
x_train = x_train.reshape()
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2-1. 모델구성1
input1 = Input(shape=(9,19)) #(N,2)
dense1 = LSTM(10,name='jk4',activation='relu')(input1) # (N,9,10)
dense2 = Dense(64,activation='relu',name='jk5')(dense1) # (N,64)
dense3 = Dense(64,activation='relu',name='jk6')(dense2) # (N,64)
output1 = Dense(1,activation='relu',name='out_ys1')(dense3)
model = Model(inputs=input1, outputs=output1)
model.summary()     


import time
start_time = time.time()
#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
earlyStopping = EarlyStopping(monitor='loss', patience=150, mode='min', 
                              verbose=1,restore_best_weights=True)

hist = model.fit(x,y,  
                epochs=3, 
                batch_size=300000,
                validation_split=0.33,
                verbose=2,
                callbacks=[earlyStopping]
                )

end_time= time.time()-start_time
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
# loss2 = model.evaluate([x1_test,x2_test,x3_test], y2_test)

print('loss :', loss)
# print('loss :', loss2)
end_time=time.time()-start_time
print('걸린 시간 :', end_time)
'''
# loss : 149.38671875
# 걸린 시간 : 51.03459930419922
