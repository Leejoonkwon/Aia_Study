import tensorflow as tf
print(tf.__version__)
# tf.compat.v1.enable_eager_execution()
print(tf.executing_eagerly()) # True
# 즉시 실행 모드는 tensorflow1.x 대에서 미동작, 
# 즉시 실행 모드는 tensorflow2.x 대에서 동작
# tensorflow 2버전에서  즉시 실행모드가 이미 동작 중이므로
# disable_eager_execution을 통해 즉시 실행 모드를 중지할 수 있다.


# 즉시 실행 모드를 끄는 명령어 
tf.compat.v1.disable_eager_execution()

print(tf.executing_eagerly()) # False 

hello = tf.constant("Hello World")

sess = tf.compat.v1.Session()

print(sess.run(hello))
