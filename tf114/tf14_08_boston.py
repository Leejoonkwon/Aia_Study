import tensorflow as tf
tf.compat.v1.set_random_seed(26)
from sklearn.datasets import load_breast_cancer,load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
#1. 데이터
datasets = load_diabetes()
x_data = datasets.data     # (442, 10)
y_data = datasets.target  # (442,)
y_data = y_data.reshape(442,1)
# print(x_data.shape,y_data.shape)

x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,train_size=0.75,shuffle=True,random_state=72)

x = tf.compat.v1.placeholder(tf.float32,shape = [None,10])
y = tf.compat.v1.placeholder(tf.float32,shape = [None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,1]),name='weight') 
# x값 열의 맞추어 행 값,y값 열의 맞추어 열 값 부여 (weight는 무조건 X값의 열에 맞게 행부여!!!)
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),name='bias')

# hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)
hypothesis = tf.compat.v1.matmul(x, w) + b
# hypothesis = tf.nn.softmax(tf.add(tf.matmul(x, w), b))

# model.add(Dense(1,activation='sigmoid',input_dim=2))
#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y)) #mse

# loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) 
#binary_crossentropy
# model.compile(loss='binary_crossentropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.3)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epoch = 300
import time
start_time = time.time()
for epochs in range(epoch):
    cost_val,h_val,_ = sess.run([loss,hypothesis,train],
                                           feed_dict={x:x_train,y:y_train})
    if epochs %10 == 0 :
        print(epochs,'\t',"loss :",cost_val,'\n',h_val)    
   
##################################### [실습]   R2로 맹그러봐

# y_predict = sess.run(tf.cast(h_val>0.5,dtype=tf.float32))
y_predict = sess.run(hypothesis,feed_dict={x:x_test,y:y_test})
from sklearn.metrics import r2_score,accuracy_score
import numpy as np
# h_val =abs(h_val)
# h_val = np.round(h_val,0)
r2 = r2_score(y_test,y_predict)
end_time = time.time()-start_time
print('r2 :', r2)
print('걸린 시간 :',end_time)
sess.close()   

# r2 : 0.7734158078067879





