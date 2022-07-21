from psutil import ZombieProcess
from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator 
import numpy as np

(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()   

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)

augument_size = 100

print(x_train[0].shape)  #  (28,28)
print(x_train[0].reshape(28*28).shape)  #  (784,)
print(np.tile(x_train[0].reshape(28*28),augument_size).reshape(-1,28,28,1).shape)
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
