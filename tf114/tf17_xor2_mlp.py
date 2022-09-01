import tensorflow as tf
tf.compat.v1.set_random_seed(123)

#1. 데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]  # (4, 2)
y_data = [[0],[1],[1],[0]]          # (4, 1)

#2. 모델 구성
#input layer 
x = tf.compat.v1.placeholder(tf.float32,shape=[None,2])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,1])
#hidden layer
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,20]))
b1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([20]))

hidden_layer1 = (tf.compat.v1.matmul(x, w1) + b1)

#output layer
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([20,1]))
b2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
# [실습 시작] 완성해 보아요

hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(hidden_layer1, w2) + b2)
# hypothesis = tf.nn.softmax(tf.add(tf.matmul(x, w), b))

# model.add(Dense(1,activation='sigmoid',input_dim=2))
#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis-y)) #mse

loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) 
#binary_crossentropy
# model.compile(loss='binary_crossentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epoch = 700
import time
start_time = time.time()
for epochs in range(epoch):
    cost_val,h_val,_ = sess.run([loss,hypothesis,train],
                                           feed_dict={x:x_data,y:y_data})
    if epochs %10 == 0 :
        print(epochs,'\t',"loss :",cost_val,'\n',h_val)    
        # print(epochs,'\t',"loss :",cost_val)    
   
##################################### [실습]   R2로 맹그러봐

y_predict = sess.run(tf.cast(h_val>=0.5,dtype=tf.float32))
from sklearn.metrics import accuracy_score
import numpy as np
# h_val =abs(h_val)
# h_val = np.round(h_val,0)
print(y_predict)
acc = accuracy_score(y_data,y_predict)
end_time = time.time()-start_time
print('acc :', acc)
print('걸린 시간 :',end_time)
sess.close()   

#  acc : 0.5
# 걸린 시간 : 3.3614895343780518



