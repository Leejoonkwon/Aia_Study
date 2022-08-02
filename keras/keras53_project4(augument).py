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
target_size = 48
xy_train= train_datagen.flow_from_directory(
    'D:\study_data\project\\train',
    target_size=(target_size,target_size),
    class_mode='categorical',
    color_mode='grayscale',
        batch_size=28709,
    shuffle=True,) # 28709
xy_test = test_datagen.flow_from_directory(
    'D:\study_data\project\\test',
    target_size=(target_size,target_size),
    class_mode='categorical',
    color_mode='grayscale',
        batch_size=10000,
    shuffle=False,) # 7178
# np.save('D:\study_data\_save\_npy\_train_x17.npy',arr=xy_test[0][0])




np.save('D:\study_data\_save\_npy\_train_x10.npy',arr=xy_train[0][0])
np.save('D:\study_data\_save\_npy\_train_y10.npy',arr=xy_train[0][1])
np.save('D:\study_data\_save\_npy\_test_x10.npy',arr=xy_test[0][0])
np.save('D:\study_data\_save\_npy\_test_y10.npy',arr=xy_test[0][1])






