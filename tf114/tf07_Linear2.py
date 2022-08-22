# y =wx + b 
import tensorflow as tf
tf.compat.v1.set_random_seed(123)

#1. 데이터
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
print(sess.run(tf.rank(x)))
'''
W = tf.Variable(1, dtype = tf.float32)
b = tf.Variable(1, dtype = tf.float32)
 
#2. 모델 구성
hypothesis = x * W + b # y = wx + b 대반전  y= w *x+b가 아니라 y =  x *w + b이다.
# hypothesis  통상 y를 이렇게 표현 

#3-1. 컴파일
loss  = tf.reduce_mean(tf.square(hypothesis - y)) 
# mse 랜덤한 기울기와 예측 값 사이에 오차를 square(제곱)해서 양수화 한다. 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# GradientDescentOptimizer 경사 하강법 
train = optimizer.minimize(loss)
# model.compile(loss='mse',optimizer = 'sgd')

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 5400
for step in range(epochs):
    sess.run(train)
    if step %30 == 0:
        print(step,sess.run(loss),sess.run(W),sess.run(b))
# 첫번째 sess.run(train)이  연산이다.sess.run(W)와(b)는 연산후 역전파 후 갱신된 W와 b의 값을 반환        
sess.close()    # 메모리 부하를 줄이기 위해 close 한다!
'''

