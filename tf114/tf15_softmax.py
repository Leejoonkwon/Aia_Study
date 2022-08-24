import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(123)

x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,1],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7],
                   ]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0],
                   ]

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
                                           feed_dict={x:x_data,y:y_data})
    if epochs %10 == 0 :
        print(epochs,'\t',"loss :",cost_val,'\n',h_val)    
   







