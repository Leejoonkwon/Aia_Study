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
target_size = 100
xy_train= train_datagen.flow_from_directory(
    'D:\study_data\project\\train',
    target_size=(target_size,target_size),
    class_mode='categorical',
        batch_size=15000,
    shuffle=True,) # 28709
xy_test = test_datagen.flow_from_directory(
    'D:\study_data\project\\test',
    target_size=(target_size,target_size),
    class_mode='categorical',
        batch_size=10000,
    shuffle=False,) # 7178
# np.save('D:\study_data\_save\_npy\_train_x17.npy',arr=xy_test[0][0])
x_data1 = np.concatenate((xy_train[0][0],xy_test[0][0])) 
y_data1 = np.concatenate((xy_train[0][1],xy_test[0][1]))

x_train,x_test,y_train,y_test = train_test_split(x_data1,
                                                 y_data1,train_size=0.8,shuffle=True,random_state=100)
# augument_size = 40000
# randidx = np.random.randint(x_train.shape[0],size=augument_size)
# x_augumented = x_train[randidx].copy()
# y_augumented = y_train[randidx].copy()
# x_data = train_datagen.flow(x_augumented,y_augumented,
#     batch_size=augument_size,
#     shuffle=False)
# x_data1 = np.concatenate((x_train,x_data[0][0])) 
# y_data1 = np.concatenate((y_train,y_augumented))
np.save('D:\study_data\_save\_npy\_train_x10.npy',arr=x_train)
np.save('D:\study_data\_save\_npy\_train_y10.npy',arr=y_train)
np.save('D:\study_data\_save\_npy\_test_x10.npy',arr=x_test)
np.save('D:\study_data\_save\_npy\_test_y10.npy',arr=y_test)






