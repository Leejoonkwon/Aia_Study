# placeholder 은 input에만 관여한다.
# 데이터를 정의하기만 한다.
import tensorflow as tf
import numpy as np
print(tf.__version__)
print(tf.executing_eagerly())
tf.compat.v1.disable_eager_execution() # 꺼
node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1,node2)


sess = tf.compat.v1.Session()
####################################요기서부터###########################
a = tf.compat.v1.placeholder(tf.float32) # float32 타입으로된 a 공간 정의
b = tf.compat.v1.placeholder(tf.float32) # float32 타입으로된 b 공간 정의
add_node = a + b


print(sess.run(add_node,feed_dict = {a:3, b:4.5}))
print(sess.run(add_node,feed_dict = {a:[1,3], b:[2,4]}))

add_and_triple = add_node * 3
# print(add_and_triple) # Tensor("mul:0", dtype=float32)

print(sess.run(add_and_triple,feed_dict = {a:3, b:4.5})) # 22.5
# print(sess.run(add_node,feed_dict = {a:[1,3], b:[2,4]}))
