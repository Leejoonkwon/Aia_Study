from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator 
import numpy as np


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
test_datagen = ImageDataGenerator(
    rescale=1./255,)
# x_train = np.load('D:\study_data\_save\_npy\_train_x1.npy')
# y_train = np.load('D:\study_data\_save\_npy\_train_y1.npy')
# x_test = np.load('D:\study_data\_save\_npy\_test_x1.npy')
# y_test = np.load('D:\study_data\_save\_npy\_test_y1.npy')
print(x_train.shape,x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape,y_test.shape) #(60000,) (10000,)

augument_size = 10
randidx = np.random.randint(x_train.shape[0],size=augument_size)

x_augumented = x_train[randidx].copy()
x_train = x_train[randidx].copy()

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28),10).reshape(-1,28,28,1), # x
    np.zeros(10) ,# y 
    batch_size=10,
    shuffle=True)
y_data = test_datagen.flow(
    np.tile(x_augumented[0].reshape(28*28),10).reshape(-1,28,28,1), # x
    np.zeros(10) ,# y 
    batch_size=10,
    shuffle=False)

xy = np.concatenate((x_data,y_data))

# [실습]
print(xy[0][0].shape) #(20, 28, 28, 1)


# 1. x_augumented 10개와 x_train 10개를 비교하는 이미지 출력할 것
import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(20):
    plt.subplot(2,10,i+1)
    plt.axis('off')
    plt.imshow(xy[0][i],cmap='gray')
    
plt.show() 
# plt.plot(hist.history['loss'],marker='.',c='red',label='loss') #순차적으로 출력이므로  y값 지정 필요 x
# plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss')





