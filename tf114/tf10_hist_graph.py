# y =wx + b 
# 실습 
# lr 수정해서 epoch를 100번 이하로 줄인다.
# step = 100 이하 , w = 1.99,b=0.99
x_train_data = [1, 2, 3]
y_train_data = [3, 5, 7]
import tensorflow as tf
tf.compat.v1.set_random_seed(123) # 랜덤시드 고정 명령어

#1. 데이터

x_train = tf.placeholder(tf.float32,shape = [None]) # input_shape가 될 것임
y_train = tf.placeholder(tf.float32,shape = [None])

W = tf.Variable(tf.random_normal([1],dtype=tf.float32))
b = tf.Variable(tf.random_normal([1],dtype=tf.float32)) # bias는 통상 0으로 잡는다.

#2. 모델 구성
hypothesis = x_train * W + b # y = wx + b 대반전  y= w *x+b가 아니라 y =  x *w + b이다.

#3-1. 컴파일
loss  = tf.reduce_mean(tf.square(hypothesis - y_train)) 
# mse 랜덤한 기울기와 예측 값 사이에 오차를 square(제곱)해서 양수화 한다. 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1766).minimize(loss)

#3-2. 훈련
# with 문으로 sess를 정의하면 close를 정의하지 않아도 구문이 끝나는 시점에서 자동 종료된다.
loss_val_list = []
W_val_list = []

with tf.compat.v1.Session() as sess:
    # sess = tf.compat.v1.Session()
    sess.run(tf.global_variables_initializer()) # 변수를 사용하기 전 모든 변수 초기화

    epochs = 500
    for step in range(epochs):
        _,loss_val,W_val,b_val=sess.run([optimizer,loss,W,b],
                 feed_dict = {x_train :x_train_data,y_train : y_train_data})
        # sess.run(train)
        if step %10 == 0:
            
            # print(step,sess.run(loss),sess.run(W),sess.run(b))
            print(step,loss_val,W_val,b_val)
        if (2.1 > W_val >=1.99) & (b_val>=0.99):
            break
        loss_val_list.append(loss_val)
        W_val_list.append(W_val)
            
    x_test = tf.compat.v1.placeholder(tf.float32,shape=[None])
    x_test_data = [6, 7, 8]        
    # x_test = [6, 7, 8]    
    y_predict = x_test * W_val + b_val # y_predict = model.predict(x_test)
    print('[6, 7, 8]의 예측',sess.run(y_predict,
                                       feed_dict = {x_test : x_test_data}))
        
    # 첫번째 sess.run(train)이  연산이다.sess.run(W)와(b)는 연산후 역전파 후 갱신된 W와 b의 값을 반환        
# sess.close()    # 메모리 부하를 줄이기 위해 close 한다!
#################################################################
import matplotlib.pyplot as plt
# plt.plot(loss_val_list)
# # plt.subplot(W_val_list)
# plt.xlabel('epochs')
# plt.ylabel('loss')

# plt.show()

plt.subplot(2, 1, 1)                # nrows=2, ncols=1, index=1
plt.plot(loss_val_list)
plt.title('1st Graph')
plt.xlabel('epochs')
plt.ylabel('loss')


plt.subplot(2, 1, 2)                # nrows=2, ncols=1, index=2
plt.plot(W_val_list)
plt.title('2nd Graph')
plt.xlabel('epochs')
plt.ylabel('Weight')

plt.tight_layout()
plt.show()


