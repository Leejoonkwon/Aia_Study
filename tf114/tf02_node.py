import tensorflow as tf 

node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
node3 = node1 + node2       # 이상태로만 하면 데이터 자료형으로 나옴

node3 = tf.add(node1,node2) # 이렇게도 가능하다.

sess = tf.compat.v1.Session()         # sess 라고 정의
print(sess.run(node3))      # 7.0
  
