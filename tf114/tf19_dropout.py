import tensorflow as tf

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,30]),name='weigts1')
b1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([30],name='bias1'))

hidden_layer =  tf.compat.v1.sigmoid(tf.matmul(x,w1) + b1)
# model.add(Dense(30, input_shape=(2,), activation='sigmoid'))
#dropout_layers = tf.compat.v1.nn.dropout(hidden_layer,keep_prob=0.7)
dropout_layers = tf.compat.v1.nn.dropout(hidden_layer,rate=0.3) 
print(hidden_layer)     # Tensor("Sigmoid:0", shape=(?, 30), dtype=float32)
print(dropout_layers)   # Tensor("dropout/mul_1:0", shape=(?, 30), dtype=float32)



