import tensorflow as tf
# print(tf.__version__)
# print('hello world')
hello = tf.constant('hello world')

# print(hello)

sess = tf.compat.v1.Session()
print(sess.run(hello))
# tensorflow'1' 는 출력할 떄 반드시 sess.run을 거쳐한다!


