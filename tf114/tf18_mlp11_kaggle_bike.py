
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten,LSTM,Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
from sklearn.svm import LinearSVC 
from sklearn.svm import LinearSVR 
from sklearn.metrics import r2_score
import numpy as np
#1. 데이터
path = 'D:\study_data\_data\_csv\kaggle_bike/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
print(train_set)

print(train_set.columns)


test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)

# sampleSubmission = pd.read_csv(path + 'sampleSubmission.csv',#예측에서 쓸거야!!
#                        index_col=0)
            
# print(test_set)
# print(test_set.shape) #(6493, 8) #train_set과 열 값이 '1'차이 나는 건 count를 제외했기 때문이다.예측 단계에서 값을 대입

# print(train_set.info()) #null은 누락된 값이라고 하고 "결측치"라고도 한다.
# print(train_set.describe()) 


###### 결측치 처리 1.중위 ##### 
print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
train_set = train_set.fillna(train_set.median())
print(train_set.isnull().sum())
print(train_set.shape)
test_set = test_set.fillna(test_set.median())

###### 결측치 처리 2.interpolate #####
# print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
# train_set = train_set.interpolate()
# print(train_set.isnull().sum())
# print(train_set.shape)
# test_set = test_set.interpolate()

####### 결측치 처리 3.mean #####
# print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
# train_set = train_set.fillna(train_set.mean())
# print(train_set.isnull().sum())
# print(train_set.shape)
# test_set = test_set.fillna(test_set.mean())

####### 결측치 처리 3.drop #####
# print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
# train_set = train_set.dropna()
# print(train_set.isnull().sum())
# print(train_set.shape)
# test_set = test_set.dropna()

def outliers(data_out):
    quartile_1, q2 , quartile_3 = np.percentile(data_out,
                                               [25,50,75]) # percentile 백분위
    print("1사분위 : ",quartile_1) # 25% 위치인수를 기점으로 사이에 값을 구함
    print("q2 : ",q2) # 50% median과 동일 
    print("3사분위 : ",quartile_3) # 75% 위치인수를 기점으로 사이에 값을 구함
    iqr =quartile_3-quartile_1  # 75% -25%
    print("iqr :" ,iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound)|
                    (data_out<lower_bound))

# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],
season_out_index= outliers(train_set['season'])[0]
holiday_out_index= outliers(train_set['holiday'])[0]
workingday_out_index= outliers(train_set['workingday'])[0]
weather_out_index= outliers(train_set['weather'])[0]
temp_out_index= outliers(train_set['temp'])[0]
atemp_out_index= outliers(train_set['atemp'])[0]
humidity_out_index= outliers(train_set['humidity'])[0]
windspeed_out_index= outliers(train_set['windspeed'])[0]
casual_out_index= outliers(train_set['casual'])[0]
registered_out_index= outliers(train_set['registered'])[0]
# print(train_set2.loc[season_out_index,'season'])
lead_outlier_index = np.concatenate((season_out_index,
                                     holiday_out_index,
                                    #  workingday_out_index,
                                     weather_out_index,
                                     temp_out_index,
                                    #  atemp_out_index,
                                    #  humidity_out_index,
                                     windspeed_out_index,
                                     casual_out_index,
                                     registered_out_index),axis=None)
print(len(lead_outlier_index)) #161개 
print(lead_outlier_index)
lead_not_outlier_index = []
for i in train_set.index:
    if i not in lead_outlier_index :
        lead_not_outlier_index.append(i)
train_set_clean = train_set.loc[lead_not_outlier_index]      
train_set_clean = train_set_clean.reset_index(drop=True)
print(train_set_clean)
x = train_set_clean.drop([ 'casual', 'registered','count'],axis=1) #axis는 컬럼 


# print(x.columns)
# print(x.shape) #(10886, 8)

y = train_set_clean['count']
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler,StandardScaler
tf.compat.v1.disable_v2_behavior()
print(x.shape,y.shape) #(10886, 8) (10886,)

y = np.array(y)
y = y.reshape(-1,1)
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.75,shuffle=True,random_state=1234)
# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x = tf.compat.v1.placeholder(tf.float32,shape = [None,8])
y = tf.compat.v1.placeholder(tf.float32,shape = [None,1])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([8,100]))
b1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100]))

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100,100]))
b2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100]))

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100,100]))
b3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100]))

w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100,1]))
b4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))

# hidden_layer 1
hidden_layer1 = (tf.compat.v1.matmul(x, w1) + b1)
hidden_layer2 = (tf.compat.v1.matmul(hidden_layer1, w2) + b2)
hidden_layer3 = tf.sigmoid(tf.compat.v1.matmul(hidden_layer2, w3) + b3)
hypothesis = (tf.compat.v1.matmul(hidden_layer3, w4) + b4)
# hypothesis = tf.nn.softmax(tf.add(tf.matmul(x, w), b))

# model.add(Dense(1,activation='sigmoid',input_dim=2))
#3-1. 컴파일
loss = tf.reduce_mean(tf.abs(hypothesis-y)) #mae
# loss = tf.reduce_mean(tf.square(hypothesis-y)) #mse

# loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) 
#binary_crossentropy
# model.compile(loss='binary_crossentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epoch = 500
import time
start_time = time.time()
for epochs in range(epoch):
    cost_val,h_val,_ = sess.run([loss,hypothesis,train],
                                           feed_dict={x:x_train,y:y_train})
    if epochs %10 == 0 :
        print(epochs,'\t',"loss :",cost_val,'\n',h_val)    
   
##################################### [실습]   R2로 맹그러봐

# y_predict = sess.run(tf.cast(h_val>0.5,dtype=tf.float32))
y_predict = sess.run(hypothesis,feed_dict={x:x_test,y:y_test})
from sklearn.metrics import r2_score,accuracy_score
import numpy as np
# h_val =abs(h_val)
# h_val = np.round(h_val,0)
r2 = r2_score(y_test,y_predict)
end_time = time.time()-start_time
print('r2 :', r2)
print('걸린 시간 :',end_time)
sess.close()  

# r2 : 0.26664655706847995
# 걸린 시간 : 40.478649854660034

#  r2 : 0.15469282549849883
# 걸린 시간 : 17.350946187973022