from psutil import ZombieProcess
from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator 
import numpy as np
import cv2

(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()   

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)
print(x_train.shape[0]) #60000
print(x_train.shape[1]) #28
print(x_train.shape[2]) #28


augument_size = 40000
randidx = np.random.randint(x_train.shape[0],size=augument_size)
print(randidx,randidx.shape) #(40000,)
print(np.min(randidx),np.max(randidx))  #0 59997
print(type(randidx)) #<class 'numpy.ndarray'>

x_augumented = x_train[randidx].copy()
y_augumented = y_train[randidx].copy()

print(x_augumented.shape)
print(y_augumented.shape)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)

x_augumented = x_augumented.reshape(x_augumented.shape[0],
                                    x_augumented.shape[1],
                                    x_augumented.shape[2], 1)

x_augumented = train_datagen.flow(x_augumented,y_augumented,
                                  batch_size=augument_size,shuffle=False).next()[0]
    
print(x_augumented,x_augumented.shape) #(40000, 28, 28, 1)
x_train = np.concatenate((x_train,x_augumented)) # 소괄호() 1개와  소괄호(()) 2개의 차이를 공부해라!
y_train = np.concatenate((y_train,y_augumented)) 
# 소괄호() 1개와  소괄호(()) 2개의 차이를 공부해라! 2개인 이유는 안에 옵션을 더 넣을 수 있기 때문이다.아무 것도 안하면 디폴트로 들어감
print(x_train.shape,y_train.shape)
'''
# .copy() 메모리에 저장하겠다.
# randint는 랜덤한 숫자 1개를 뽑는 모듈 
# ex) randint(9)    0~8 까지 중 임의 숫자 1개 선택
# ex) randint(1,20) 1~19까지 중 임의 숫자 1개 선택
#  (28,28)
print(x_train[0].reshape(28*28).shape)  #  (784,)
print(np.tile(x_train[0].reshape(28*28),augument_size).reshape(-1,28,28,1).shape)  #(40000, 28, 28, 1)
# np.tile 의 기능은 쌓는 것이다. (x,y)라면 y만큼 x를 반복하며 쌓는 것이다.

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28),augument_size).reshape(-1,28,28,1), # x
    np.zeros(augument_size) ,# y 
    batch_size=augument_size,
    shuffle=True)#.next()

##################next() 를 쓸 경우########################
# x_data[0]                   첫번째 배치값
# x_data[0][0]                x 값
# x_data[0][0][1]             y 값
####################위 구조에서 next를 쓰면 첫 번째 []를 생략하고 진행한다.

##################next() 를 쓰지 않을 경우 #################
# x_data[0][0]              첫번째 배치값
# x_data[0][0][0]           x 값
# x_data[0][0][0][1]        y 값


# print(np.zeros(augument_size))
# print(np.zeros(augument_size).shape) #(100,)
# print(x_data) #<keras.preprocessing.image.NumpyArrayIterator object at 0x0000017003255B80>
# print(x_data[0][0].shape)  #(100, 28, 28, 1) x 값
# print(x_data[0][1].shape)  #(100,) y 값


import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.axis('off')
    plt.imshow(x_data[0][0][i],cmap='gray')
plt.show() 
'''