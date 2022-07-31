from keras.preprocessing.image import ImageDataGenerator 
import numpy as np
from sklearn.model_selection import train_test_split



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
    
# x_augument = x_train[randidx].copy()
# y_augument = y_train[randidx].copy()
# x_10_train = x_train[randidx]
# print(x_augument.shape)

      
xy_train= train_datagen.flow_from_directory(
    'D:\study_data\\train',
    target_size=(100,100),
    class_mode='categorical',
    color_mode='grayscale',
    batch_size=2385,
    shuffle=False,) # 경로 및 폴더 설정
x = xy_train[0][0]
y = xy_train[0][1]
print(y.shape)
print(x.shape)

augument_size = 2385

randidx = np.random.randint(x.shape[0],size=augument_size)
x_augumented = x[randidx].copy()
y_augumented = y[randidx].copy()
# print(x_augumented.shape) #(164, 48, 48, 1)
# print(y_augumented.shape) #(164, 21)

x_data = train_datagen.flow(
    np.tile(x_augumented[0].reshape(100*100*1),40000).reshape(-1,100,100,1), # x
    np.tile(y_augumented[0].reshape(21*1),40000).reshape(-1,21) ,# y 
    batch_size=40000,
    shuffle=True)


x_data1 = np.concatenate((x_data[0][0],x)) # 48872 행이네 
y_data1 = np.concatenate((x_data[0][1],y))
# print(x_data1.shape)    # (22385, 100, 100, 1)
# print(y_data1.shape)    # (22385, 21)
# [실습]

x_train,x_test,y_train,y_test = train_test_split(x_data1,y_data1,train_size=0.8,shuffle=True,random_state=100)
np.save('D:\study_data\_save\_npy\_train_x10.npy',arr=x_train)
np.save('D:\study_data\_save\_npy\_train_y10.npy',arr=y_train)
np.save('D:\study_data\_save\_npy\_test_x10.npy',arr=x_test)
np.save('D:\study_data\_save\_npy\_test_y10.npy',arr=y_test)



# # 1. x_augumented 10개와 x_train 10개를 비교하는 이미지 출력할 것
# import matplotlib.pyplot as plt
# plt.figure(figsize=(7,7))
# for i in range(200):
#     plt.subplot(20,10,i+1)
#     plt.axis('off')
#     plt.imshow(x_data[i],cmap='gray')
    
# plt.show() 


