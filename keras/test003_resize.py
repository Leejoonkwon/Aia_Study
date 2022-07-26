from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator 
import numpy as np



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
    rescale=1./255,
   
)
xy_train= train_datagen.flow_from_directory(
    'D:\study_data\_data\emotion\\train',
    target_size=(150,150),
    class_mode='categorical',
    batch_size=28709,
    shuffle=True,) # 경로 및 폴더 설정
x_train = xy_train[0][0]

augument_size = 10
randidx = np.random.randint(x_train.shape[0],size=augument_size)

# print(randidx,randidx.shape) #((20,)

# print(np.min(randidx),np.max(randidx)) # 174 49920
# print(type(randidx)) #<class 'numpy.ndarray'>

x_augumented = x_train[randidx].copy()
x_train = x_train[randidx].copy()


# 1. x_augumented 10개와 x_train 10개를 비교하는 이미지 출력할 것
import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(20):
    plt.subplot(2,10,i+1)
    plt.axis('off')
    plt.imshow(xy_train[0][i],cmap='gray')
    
plt.show() 
