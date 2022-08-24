import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(123)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
import pandas as pd
datasets = load_iris()
x_data = datasets.data      
y_data = datasets.target     
# y_data = y_data.reshape(150,1)
y_data = pd.get_dummies(y_data)
# print(x_data.shape,y_data.shape)  # (150, 4) (150,3)
# print(np.unique(y_data,return_counts=True)) #(array([0, 1, 2]), array([50, 50, 50], dtype=int64))
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,train_size=0.8,random_state=1234)
x = tf.compat.v1.placeholder(tf.float32,shape=[None,4])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([4,3]))

b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1,3]))

y = tf.compat.v1.placeholder(tf.float32,shape=[None,3])
#2. 모델구성 // 시작

hypothesis = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(x,w) + b)
# model.add(Dense(3,activation='softmax',input_dim=4))

#3-1. 컴파일
loss = tf.compat.v1.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))
# model.compile(loss='categorical_crossentropy)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(loss)
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# [실습]
# 맹그러
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epoch = 2000
import time
start_time = time.time()
for epochs in range(epoch):
    cost_val,h_val,_ = sess.run([loss,hypothesis,train],
                                           feed_dict={x:x_train,y:y_train})
    if epochs %10 == 0 :
        print(epochs,'\t',"loss :",cost_val,'\n',h_val)    
   
y_predict = sess.run(hypothesis,feed_dict={x:x_test,y:y_test})
y_predict = np.argmax(y_predict,axis=1)
# y_test = np.argmax(y_test,axis=1)

# y_predict = pd.get_dummies(y_predict)
# print(y_predict)
print(type(y_test))

# acc = accuracy_score(y_test,y_predict)
# print('acc :',acc)

# acc : 0.9666666666666667


