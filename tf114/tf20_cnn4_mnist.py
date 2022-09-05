import tensorflow as tf
import keras
import numpy as np
from sklearn.metrics import accuracy_score
tf.compat.v1.set_random_seed(123)
# tf.executing_eagerly()
tf.compat.v1.disable_eager_execution()

#1. 데이터
from keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape) #  (60000, 10)
x_train = x_train.reshape(60000,28,28,1).astype('float32')/255. # MinMaxScaling
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

#2. 모델구성
x = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=([None, 28, 28, 1])) # input_shape
y = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=([None, 10]))
'''
# model.add(Conv2D(filters=64, kernel_size=(3, 3),   # 출력(4,4,10)                                    
#                  padding='same',
#                  input_shape=(28, 28, 1)))    #(batch_size, row, column, channels)     
                                                                                           
# model.add(MaxPooling2D())

#  #    (kernel_size * channls) * filters = summary Param 개수(CNN모델)  
# model.add(Conv2D(32, (2,2),  #인풋쉐이프에 행값은 디폴트는 32
#                  padding = 'same',         # 디폴트값(안준것과 같다.) 
#                  activation= 'swish'))    # 출력(3,3,7)       
# model.add(MaxPooling2D())
# model.add(Conv2D(100, (2,2), 
#                  padding = 'same',         # 디폴트값(안준것과 같다.) 
#                  activation= 'swish'))    # 출력(3,3,7)      
                                              
# model.add(Flatten()) # (N, 63) 위치와 순서는 바뀌지 않아야한다.transpose와 전혀 다르다.
model.add(Dense(100,input_shape=(784,)))
model.add(Dense(100,activation='swish'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='swish'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='sigmoid'))
'''
# Layer1
w1 = tf.compat.v1.get_variable('w1', shape=[3, 3, 1, 64]) 
# kernel_size=(2,2),
# color=1,  다음 레이어에서는 인풋 쉐이프가 됨 tensorflow2에서는 표기하지 않았지만 
# 자동으로 계산되는 것이였음
# filters=64 output
# L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='VALID')
L1 = tf.compat.v1.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.compat.v1.nn.relu(L1)
L1_maxpool = tf.compat.v1.nn.max_pool2d(L1,ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
# model.add(Conv2d(64, kernel_size=(2,2),input=(28,28,1),activation='relu'))

print(w1)           # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
print(L1)           # Tensor("Relu:0", shape=(?, 28, 28, 128), dtype=float32)
print(L1_maxpool)   # Tensor("MaxPool2d:0", shape=(?, 14, 14, 128), dtype=float32)

# Layer2

w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 64, 32]) 
# kernel_size=(2,2),
# color=1,  다음 레이어에서는 인풋 쉐이프가 됨 tensorflow2에서는 표기하지 않았지만 
# 자동으로 계산되는 것이였음
# filters=64 output
# L1 = tf.nn.conv2d(x, w2, strides=[1, 1, 1, 1], padding='VALID')
L2 = tf.compat.v1.nn.conv2d(L1_maxpool, w2, strides=[1, 1, 1, 1], padding='VALID')
L2 = tf.compat.v1.nn.relu(L2)
L2_maxpool = tf.nn.max_pool2d(L2,ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')

print(L2)           # Tensor("Selu:0", shape=(?, 12, 12, 64), dtype=float32)
print(L2_maxpool)   # Tensor("MaxPool2d_1:0", shape=(?, 6, 6, 64), dtype=float32)

# Layer3

w3 = tf.compat.v1.get_variable('w3', shape=[3, 3, 32, 100]) 
# kernel_size=(2,2),
# color=1,  다음 레이어에서는 인풋 쉐이프가 됨 tensorflow2에서는 표기하지 않았지만 
# 자동으로 계산되는 것이였음
# filters=64 output
# L1 = tf.nn.conv2d(x, w3, strides=[1, 1, 1, 1], padding='VALID')
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1, 1, 1, 1], padding='VALID')
L3 = tf.nn.relu(L3)
# L3_maxpool = tf.nn.max_pool2d(L3,ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')

print(L3)           # Tensor("Elu:0", shape=(?, 4, 4, 32), dtype=float32)

# flatten layer
L_flatten = tf.reshape(L3,[-1,4*4*100])
print("플래튼 :",L_flatten) # 플래튼 : Tensor("Reshape:0", shape=(?, 512), dtype=float32)

# Layer4 Dense
w4 = tf.compat.v1.get_variable('w4',shape=[4*4*100,100])

b4 = tf.compat.v1.get_variable('b4',shape =[100])
L4 = tf.nn.relu(tf.matmul(L_flatten,w4) + b4)
L4 = tf.nn.dropout(L4, rate=0.3) # rate = 0.3

# Layer5 Dense
w5 = tf.compat.v1.get_variable('w5',shape=[100,10])

b5 = tf.compat.v1.get_variable('b5',shape =[10])
L5 = tf.matmul(L4,w5) + b5
hypothesis = tf.nn.softmax(L5) # Tensor("Softmax:0", shape=(?, 10), dtype=float32)
print(hypothesis) # Tensor("Softmax:0", shape=(?, 10), dtype=float32)


#3-1. 컴파일
# loss = tf.compat.v1.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))
loss = tf.compat.v1.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis,labels=y))
# model.compile(loss='categorical_crossentropy)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(loss)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
train_epoch = 30
batch_size = 1000
total_batch = int(len(x_train)/batch_size)
print(total_batch) # 600

import time
start_time = time.time()
for epoch in range(train_epoch):   # 총 30번 돈다
    avg_loss = 0
    for i in range(total_batch):
        start = i* batch_size       # 0
        end = start+ batch_size     # 100
        batch_x,batch_y = x_train[start:end],y_train[start:end] # 0~ 100
        feed_dict = {x:batch_x, y:batch_y}
        batch_loss,_ = sess.run([loss, optimizer],
                                           feed_dict=feed_dict)
        avg_loss += batch_loss/total_batch
    print('Epoch :','%04d'%(epoch +1),'loss :{:.9f}'.format(avg_loss))

prediction = tf.compat.v1.equal(tf.compat.v1.arg_max(hypothesis, 1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction,tf.float32))
print("ACC :",sess.run(accuracy,feed_dict={x:x_test,y:y_test}))
# acc : 0.1277
# Adam으로 optimizer
# acc : 0.813





