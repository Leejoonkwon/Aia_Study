from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator 
import numpy as np



train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(
    rescale=1./255,)
    
augument_size = 10              # 300장  +300        ,10000 
randidx = np.random.randint(x_train.shape[0], size=augument_size)

x_augument = x_train[randidx].copy()
y_augument = y_train[randidx].copy()
x_10_train = x_train[randidx]
print(x_augument.shape)


xy_train= train_datagen.flow_from_directory(
    'D:\study_data\\train',
    target_size=(150,150),
    class_mode='categorical',
    color_mode='grayscale',
    batch_size=28872,
    shuffle=False,) # 경로 및 폴더 설정
x = xy_train[0][0]
y = xy_train[0][1]
print(y.shape)
np.save('D:\study_data\_save\_npy\\train_x.npy',arr=x)
np.save('D:\study_data\_save\_npy\\train_y.npy',arr=y)

'''
x = np.load('D:\study_data\_save\_npy\\train_x.npy')
y = np.load('D:\study_data\_save\_npy\\train_y.npy')

print(x.shape) #(28872, 150, 150, 1)
print(y.shape) #(28872, 21)

# print(xy_train[0][0].shape) #(28872, 150, 150, 1)

augument_size = 100
x1 = x[:28708] # kaggle data
x2 = x[28708:] # 직접 모은 데이터
y1 = y[:28708] # kaggle data
y2 = y[28708:] # 직접 모은 데이터
print(x1.shape,x2.shape) #(28708, 150, 150, 1) (164, 150, 150, 1)
print(y1.shape,y2.shape) #(28708, 21) (164, 21)

augument_size = 162

randidx = np.random.randint(x2.shape[0],size=augument_size)
x_augumented = x2[randidx].copy()
y_augumented = y2[randidx].copy()
x2 = x2[randidx]
y2 = y2[randidx]
x_data = train_datagen.flow(
    np.tile(x_augumented[0].reshape(150*150),20000).reshape(-1,150,150,1), # x
    np.zeros(20000) ,# y 
    batch_size=20000,
    shuffle=True)


print(x_data[0][0].shape) # (20000, 150, 150, 1)
print(x_data[0][1].shape) # (20000,)

print(x.shape) #(28872, 150, 150, 1)
print(y.shape) #(28872, 21)


x_data1 = np.concatenate((x,x_data[0][0])) # 48872 행이네 
# y_data1 = np.concatenate((y,x_data[0][1]))
# print(x_data1.shape)# (510, 150, 150, 1)
# print(y_data1.shape) #(1000, 150, 150, 1)
# [실습]


# 1. x_augumented 10개와 x_train 10개를 비교하는 이미지 출력할 것
import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(200):
    plt.subplot(20,10,i+1)
    plt.axis('off')
    plt.imshow(x_data[i],cmap='gray')
    
plt.show() 
# plt.plot(hist.history['loss'],marker='.',c='red',label='loss') #순차적으로 출력이므로  y값 지정 필요 x
# plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss')
'''


