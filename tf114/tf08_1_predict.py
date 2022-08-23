# y =wx + b 
import tensorflow as tf
tf.compat.v1.set_random_seed(123) # 랜덤시드 고정 명령어

#1. 데이터
# x = [1, 2, 3, 4, 5]
# y = [1, 2, 3, 4, 5]
x_train = tf.placeholder(tf.float32,shape = [None]) # input_shape가 될 것임
y_train = tf.placeholder(tf.float32,shape = [None])
# tensorflow 1 은 모델 레이어마다 쉐이프를 잡아줘야한다.(떡밥)

# W = tf.Variable(1, dtype = tf.float32)
# b = tf.Variable(1, dtype = tf.float32)
W = tf.Variable(tf.random_normal([1],dtype=tf.float32))
b = tf.Variable(tf.random_normal([1],dtype=tf.float32)) # bias는 통상 0으로 잡는다.
# normal() 함수
# 0~1 사이의 값 호출
# random_normal  # Random float x, 0.0 <= x <= 1.0
# uniform() 함수
# 2개의 숫자 사이의 랜덤 실수를 리턴합니다.
#  random.uniform(1, 10)  # Random float x, 1.0 <= x < 10.0
# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer()) 
# print(sess.run(W)) # [-1.5080816]

#2. 모델 구성
hypothesis = x_train * W + b # y = wx + b 대반전  y= w *x+b가 아니라 y =  x *w + b이다.
# hypothesis  통상 y를 이렇게 표현 

#3-1. 컴파일
loss  = tf.reduce_mean(tf.square(hypothesis - y_train)) 
# mse 랜덤한 기울기와 예측 값 사이에 오차를 square(제곱)해서 양수화 한다. 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
# GradientDescentOptimizer 경사 하강법 
# train = optimizer.minimize(loss) # 최솟값 찾기
# model.compile(loss='mse',optimizer = 'sgd')


#3-2. 훈련
# with 문으로 sess를 정의하면 close를 정의하지 않아도 구문이 끝나는 시점에서 자동 종료된다.
with tf.compat.v1.Session() as sess:
    # sess = tf.compat.v1.Session()
    sess.run(tf.global_variables_initializer()) # 변수를 사용하기 전 모든 변수 초기화

    epochs = 2001
    for step in range(epochs):
        _,loss_val,W_val,b_val=sess.run([optimizer,loss,W,b],
                 feed_dict = {x_train :[1,2,3,4,5],y_train : [1,2,3,4,5]})
        # sess.run(train)
        if step %30 == 0:
            # print(step,sess.run(loss),sess.run(W),sess.run(b))
            print(step,loss_val,W_val,b_val)
            
    x_test = tf.compat.v1.placeholder(tf.float32,shape=[None])
            
    # x_test = [6, 7, 8]    
    y_predict = x_test * W_val + b_val # y_predict = model.predict(x_test)
    print('[6, 7, 8]의 예측',sess.run(y_predict,
                                       feed_dict = {x_test : [6, 7, 8]}))
        
    # 첫번째 sess.run(train)이  연산이다.sess.run(W)와(b)는 연산후 역전파 후 갱신된 W와 b의 값을 반환        
# sess.close()    # 메모리 부하를 줄이기 위해 close 한다!
#################################################################

