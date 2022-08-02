from keras.preprocessing.image import ImageDataGenerator 
import numpy as np
from sklearn.model_selection import train_test_split
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=5,
    zoom_range=0.1,
    fill_mode='nearest')
test_datagen = ImageDataGenerator(
    rescale=1./255,)
xy_train= test_datagen.flow_from_directory(
    'D:\study_data\\train',
    target_size=(100,100),
    class_mode='categorical',
    color_mode='grayscale',
    batch_size=2385,
    shuffle=True,) # 경로 및 폴더 설정
xy_test = test_datagen.flow_from_directory(
    'D:\study_data\project',
    target_size=(100,100),
    class_mode='categorical',
    color_mode='grayscale',
    batch_size=2385,
    shuffle=False,)
np.save('D:\study_data\_save\_npy\_train_x17.npy',arr=xy_test[0][0])

'''
x_train,x_test,y_train,y_test = train_test_split(xy_train[0][0],
                                                 xy_train[0][1],train_size=0.8,shuffle=True,random_state=100)
augument_size = 40000
randidx = np.random.randint(x_train.shape[0],size=augument_size)
x_augumented = x_train[randidx].copy()
y_augumented = y_train[randidx].copy()
x_data = train_datagen.flow(x_augumented,y_augumented,
    batch_size=augument_size,
    shuffle=False)
x_data1 = np.concatenate((x_train,x_data[0][0])) 
y_data1 = np.concatenate((y_train,y_augumented))
np.save('D:\study_data\_save\_npy\_train_x15.npy',arr=x_data1)
np.save('D:\study_data\_save\_npy\_train_y15.npy',arr=y_data1)
np.save('D:\study_data\_save\_npy\_test_x15.npy',arr=x_test)
np.save('D:\study_data\_save\_npy\_test_y15.npy',arr=y_test)
'''





