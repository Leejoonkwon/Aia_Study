# [실습]
# DNN으로 구성
import tensorflow as tf
import keras
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
tf.compat.v1.set_random_seed(123)
'''
model.add(Dense(100,input_shape=(784,),activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10,activation='softmax'))
'''

#1. 데이터
from keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape) #  (60000, 10)
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)

x = tf.compat.v1.placeholder(tf.float32,shape=([None,784]))
y = tf.compat.v1.placeholder(tf.float32,shape=([None, 10]))

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([784,100]))
b1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100]))
hidden_layer =  (tf.matmul(x,w1) + b1)
dropout_layers = tf.compat.v1.nn.dropout(hidden_layer,rate=0.25) 

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100,100]))
b2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100]))
hidden_layer2 =  tf.nn.relu(tf.matmul(dropout_layers,w2) + b2)
dropout_layers2 = tf.compat.v1.nn.dropout(hidden_layer2,rate=0.25) 

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100,100]))
b3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100]))
hidden_layer3 =  tf.nn.sigmoid(tf.matmul(dropout_layers2,w3) + b3)
dropout_layers3= tf.compat.v1.nn.dropout(hidden_layer3,rate=0.25) 

w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100,10]))
b4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10]))
hypothesis =  tf.compat.v1.nn.softmax(tf.matmul(dropout_layers3,w4) + b4)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)
#3-1. 컴파일
loss = tf.compat.v1.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))
# model.compile(loss='categorical_crossentropy)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(loss)
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

# [실습]
# 맹그러

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epoch = 200
import time
start_time = time.time()
for epochs in range(epoch):
    cost_val,h_val,_ = sess.run([loss,hypothesis,train],
                                           feed_dict={x:x_train,y:y_train})
    if epochs %10 == 0 :
        print(epochs,'\t',"loss :",cost_val,'\n',h_val)    
   
y_predict = sess.run(hypothesis,feed_dict={x:x_test,y:y_test})
# y_predict = np.argmax(y_predict,axis=1)
# y_test = np.argmax(y_test,axis=1)

# y_predict = pd.get_dummies(y_predict)
# print(y_predict)

y_test=np.argmax(np.array(y_test),axis=1)
y_predict=np.argmax(y_predict,axis=1)

acc = accuracy_score(y_test,y_predict)
print('acc :',acc)

# acc : 0.1277
# Adam으로 optimizer
# acc : 0.813
