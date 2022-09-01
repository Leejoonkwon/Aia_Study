import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(26)
from sklearn.datasets import load_iris,load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
datasets = load_wine()
x_data = datasets.data      
y_data = datasets.target     
y_data = pd.get_dummies(y_data)
# y_data =y_data.reshape(-1, 1)
# print(y_data)

print(x_data.shape,y_data.shape)  # (178, 13) (178, 3)
# print(np.unique(y_data,return_counts=True)) #(array([0, 1, 2]), array([59, 71, 48], dtype=int64))

x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,train_size=0.8,random_state=1234)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)
'''
model.add(Conv2D(filters=64, kernel_size=(1, 1),   # 출력(4,4,10)                                    
                 padding='same',
                 input_shape=(13, 1,1)))    #(batch_size, row, column, channels)     
                                                                                           

 #    (kernel_size * channls) * filters = summary Param 개수(CNN모델)  
model.add(Conv2D(32, (1,1),  #인풋쉐이프에 행값은 디폴트는 32
                 padding = 'same',         # 디폴트값(안준것과 같다.) 
                 activation= 'swish'))    # 출력(3,3,7)       
model.add(Conv2D(64, (1,1), 
                 padding = 'same',         # 디폴트값(안준것과 같다.) 
                 activation= 'swish'))    # 출력(3,3,7)      
model.add(Flatten())  
model.add(Dense(100,input_dim=13))
# model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))
'''
x = tf.compat.v1.placeholder(tf.float32,shape=[None,13])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,3])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([13,100]))
b1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100]))

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100,100]))
b2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100]))

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100,100]))
b3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100]))

w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100,3]))
b4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([3]))
# hidden_layer 1
hidden_layer1 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(x, w1) + b1)
hidden_layer2 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(hidden_layer1, w2) + b2)
hidden_layer3 = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(hidden_layer2, w3) + b3)
#2. 모델구성 // 시작

hypothesis = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(hidden_layer3,w4) + b4)
# model.add(Dense(3,activation='softmax',input_dim=4))

#3-1. 컴파일
# loss = tf.compat.v1.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))
loss = tf.compat.v1.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))
# model.compile(loss='categorical_crossentropy)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(loss)
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# [실습]
# 맹그러
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epoch = 500
import time
start_time = time.time()
for epochs in range(epoch):
    cost_val,h_val,_ = sess.run([loss,hypothesis,train],
                                           feed_dict={x:x_train,y:y_train})
    if epochs %10 == 0 :
        # print(epochs,'\t',"loss :",cost_val,'\n',h_val)    
        print(epochs,'\t',"loss :",cost_val)
   
y_predict = sess.run(hypothesis,feed_dict={x:x_test,y:y_test})


# y_predict = pd.get_dummies(y_predict)
y_predict = np.argmax(y_predict,axis=1)
y_test = y_test.values
y_test = np.argmax(y_test,axis=1)
print(y_predict.shape)
print(y_test.shape)
# y_predict = tf.keras.utils.to_categorical(y_predict)
acc = accuracy_score(y_test,y_predict)
print('acc :',acc)

# acc : 0.8333333333333334

