import tensorflow as tf
tf.compat.v1.set_random_seed(26)
from sklearn.datasets import load_breast_cancer,load_diabetes,load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
#1. 데이터
datasets = load_boston()
x_data = datasets.data       
y_data = datasets.target   
y_data = y_data.reshape(-1, 1)
print(x_data.shape,y_data.shape) #(506, 13) (506,)

x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,train_size=0.75,shuffle=True,random_state=72)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
'''
model.add(Conv1D(10,2,input_shape=(13,1)))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
'''
x = tf.compat.v1.placeholder(tf.float32,shape = [None,13])
y = tf.compat.v1.placeholder(tf.float32,shape = [None,1])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([13,100]))
b1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100]))

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100,100]))
b2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100]))

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100,100]))
b3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100]))

w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100,1]))
b4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
# hidden_layer 1
hidden_layer1 = (tf.compat.v1.matmul(x, w1) + b1)
hidden_layer2 = (tf.compat.v1.matmul(hidden_layer1, w2) + b2)
hidden_layer3 = tf.sigmoid(tf.compat.v1.matmul(hidden_layer2, w3) + b3)
hypothesis = (tf.compat.v1.matmul(hidden_layer3, w4) + b4)

# model.add(Dense(1,activation='sigmoid',input_dim=2))
#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y)) #mse

# loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) 
#binary_crossentropy
# model.compile(loss='binary_crossentropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
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

# r2 : 0.6887545284527438
# 걸린 시간 : 5.295245170593262



