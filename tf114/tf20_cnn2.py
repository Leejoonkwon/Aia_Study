import tensorflow as tf
import numpy as  np
tf.compat.v1.set_random_seed(123)

#1 . 데이터
x_train = np.array([[[[1],[2],[3]],
                    [[4],[5],[6]],
                    [[7],[8],[9]],
                    ]])
print(x_train.shape) # (1, 3, 3, 1)

x = tf.compat.v1.placeholder(tf.float32,shape=[None,3,3,1])

w1 = tf.compat.v1.constant([[[[1.]],[[1.]]],
                             [[[1.]],[[1.]]]])

print(w1.shape)

# L1 = tf.nn.conv2d(x, w1 ,strides=(1,1,1,1),padding='VALID')
# print(L1)+
# L1 = tf.nn.conv2d(x, w1 ,strides=(1,1,1,1),padding='SAME')
L1 = tf.nn.conv2d(x, w1 ,strides=(1,2,2,1),padding='SAME')

sess =  tf.compat.v1.Session()
output = sess.run(L1,feed_dict={x:x_train})
print("===============결과===========")
print(output)
print("========결과 shape=====")
print(output.shape) # (1, 2, 2, 1)










































