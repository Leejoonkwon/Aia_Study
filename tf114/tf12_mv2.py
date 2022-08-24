import tensorflow as tf
tf.compat.v1.set_random_seed(26)

x_data=[[73, 51, 65],
        [92, 98, 11],
        [89, 31, 33],
        [99, 33, 100],
        [17, 66, 79]
        ] # (5, 3)
y_data = [[152],[185],[180],[205],[142]] #(5, )

x = tf.compat.v1.placeholder(tf.float32,shape = [None,3])
y = tf.compat.v1.placeholder(tf.float32,shape = [None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,1]),name='weight') 
# x값 열의 맞추어 행 값,y값 열의 맞추어 열 값 부여 (weight는 무조건 X값의 열에 맞게 행부여!!!)
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),name='bias')

hypothesis = tf.compat.v1.matmul(x, w) + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y)) #mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00008108)
train = optimizer.minimize(loss)

w_history = []
loss_history = []

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epoch = 500000
import time
start_time = time.time()
for epochs in range(epoch):
    cost_val,h_val,_ = sess.run([loss,hypothesis,train],
                                           feed_dict={x:x_data,y:y_data})
    if epochs %10000 == 0 :
        print(epochs,'\t',"loss :",cost_val,'\n',h_val)    
   
##################################### [실습]   R2로 맹그러봐


# print(y_predict)
from sklearn.metrics import r2_score
r2 = r2_score(y_data,h_val)
end_time = time.time()-start_time
print('r2 :', r2)
print('걸린 시간 :',end_time)
sess.close()   

# r2 : 0.7734158078067879
# 걸린 시간 : 105.03490161895752



