from numpy import gradient
import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.set_random_seed(123)
x_train = [1]
y_train = [1]
x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),name = 'weight')
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
print(sess.run(w))

hypothesis = x*w

# loss = tf.reduce_mean(tf.square(hypothesis-y)) #mse
loss = tf.reduce_mean(tf.abs(hypothesis-y)) #mae

#==========================================================| GradientDescentOptimizer
lr = 0.1                                                  #| 풀어쓴 것 
gradient = tf.reduce_mean((w * x - y ) * x)               #|
descent = w - lr * gradient                               #|
update = w.assign(descent)                                #|
#==========================================================|

w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(21):
    _,loss_v,w_v = sess.run([update,loss,w],feed_dict={x:x_train,y:y_train})
    print(step,'\t',loss_v,'\t',w_v)    
    w_history.append(w_v)
    loss_history.append(loss_v)
    
# print('====================W_history=================')        
# print(w_history)
# print('====================Loss_history=================')        
# print(loss_history)




# plt.plot(w_history,loss_history)
# plt.xlabel('Weight')
# plt.ylabel('Loss')
# plt.show()

