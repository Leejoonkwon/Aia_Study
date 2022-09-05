import tensorflow as tf
import keras
import numpy as np

tf.compat.v1.set_random_seed(123)

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
x = tf.compat.v1.placeholder(tf.float32, shape=([None, 28, 28, 1])) # input_shape
y = tf.compat.v1.placeholder(tf.float32, shape=([None, 10]))

w1 = tf.compat.v1.get_variable('w1', shape=[2, 2, 1, 64]) 
# kernel_size=(2,2),
# color=1,  다음 레이어에서는 인풋 쉐이프가 됨 tensorflow2에서는 표기하지 않았지만 
# 자동으로 계산되는 것이였음
# filters=64 output
L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='VALID')
# L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
# model.add(Conv2d(64, kernel_size=(2,2),input=(28,28,1)))

print(w1) # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
print(L1) # Tensor("Conv2D:0", shape=(?, 27, 27, 64), dtype=float32)




