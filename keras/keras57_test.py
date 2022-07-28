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

      
# xy_train= train_datagen.flow_from_directory(
#     'D:\study_data\\train',
#     target_size=(48,48),
#     class_mode='categorical',
#     color_mode='grayscale',
#     batch_size=28872,
#     shuffle=False,) # 경로 및 폴더 설정
# x = xy_train[0][0]
# y = xy_train[0][1]
# print(y.shape)
# np.save('D:\study_data\_save\_npy\\train_x.npy',arr=x)
# np.save('D:\study_data\_save\_npy\\train_y.npy',arr=y)


x = np.load('D:\study_data\_save\_npy\\train_x.npy')
y = np.load('D:\study_data\_save\_npy\\train_y.npy')

print(x.shape) #(28872, 48, 48, 1)
print(y.shape) #(28872, 21)

# print(xy_train[0][0].shape) #(28872, 150, 150, 1)

x2 = x[28708:] # 직접 모은 데이터
y2 = y[28708:] # 직접 모은 데이터

augument_size = 164

randidx = np.random.randint(x2.shape[0],size=augument_size)
x_augumented = x2[randidx].copy()
y_augumented = y2[randidx].copy()
print(x_augumented.shape) #(164, 21)
print(y_augumented.shape) #(164, 21)
'''
x2 = x2[randidx]
y2 = y2[randidx]
x_data = train_datagen.flow(
    np.tile(x_augumented[0].reshape(150*150),20000).reshape(-1,150,150,1), # x
    np.tile(y_augumented[0].reshape(21*1),20000).reshape(-1,21) ,# y 
    batch_size=20000,
    shuffle=True)


print(x_data[0][0].shape) # (20000, 150, 150, 1)
print(x_data[0][1].shape) # (20000, 21)

print(x.shape) #(28872, 150, 150, 1)
print(y.shape) #(28872, 21)


x_data1 = np.concatenate((x,x_data[0][0])) # 48872 행이네 
y_data1 = np.concatenate((y,x_data[0][1]))
print(x_data1.shape)# (48872, 150, 150, 1)
print(y_data1.shape) #(48872, 21)
# [실습]
x_train,x_test,y_train,y_test = train_test_split(x_data1,y_data1,train_size=0.7,shuffle=False)
np.save('D:\study_data\_save\_npy\_train_x4.npy',arr=x_train)
np.save('D:\study_data\_save\_npy\_train_y4.npy',arr=y_train)
np.save('D:\study_data\_save\_npy\_test_x4.npy',arr=x_test)
np.save('D:\study_data\_save\_npy\_test_y4.npy',arr=y_test)



# 1. x_augumented 10개와 x_train 10개를 비교하는 이미지 출력할 것
import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(200):
    plt.subplot(20,10,i+1)
    plt.axis('off')
    plt.imshow(x_data[i],cmap='gray')
    
plt.show() 
'''

