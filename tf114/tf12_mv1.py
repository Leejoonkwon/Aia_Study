from pickletools import optimize
from tkinter import E
import tensorflow as tf
tf.compat.v1.set_random_seed(1234)

#1. 데이터
x1_data = [73., 93.,89.,96.,73.]
x2_data = [80., 88.,91.,98.,66.]
x3_data = [75., 93.,90.,100.,70.]
y_data = [152., 185.,180.,196.,142.]

x1 = tf.compat.v1.placeholder(tf.float32)
x2 = tf.compat.v1.placeholder(tf.float32)
x3 = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1],dtype=tf.float32))
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1],dtype=tf.float32))
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1],dtype=tf.float32))
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1],dtype=tf.float32))

#2. 모델
hypothesis = x1*w1 + x2*w2 + x3*w3 + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y)) #mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(loss)

w_history = []
loss_history = []

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(100):
    cost_val,h_val,_ = sess.run([loss,hypothesis,train],
                                           feed_dict={x1:x1_data,x2:x2_data,x3:x3_data,y:y_data})
    if epochs %20 == 0 :
        print(epochs,'\t',"loss :",cost_val,'\n',h_val)    
    # w_history.append(w_v)
    # loss_history.append(loss_v)
##################################### [실습]   R2로 맹그러봐
# y_predict = x1_data * w1_v + x2_data * w2_v + x3_data * w3_v + b_v


# print(y_predict)
from sklearn.metrics import r2_score
r2 = r2_score(y_data,h_val)
print('r2 :', r2)
sess.close()   


