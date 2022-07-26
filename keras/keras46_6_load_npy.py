import numpy as np      
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn import datasets

# np.save('D:/study_data/_save/_npy/keras46_5_train_x.npy',arr=xy_train[0][0])
# np.save('D:/study_data/_save/_npy/keras46_5_train_y.npy',arr=xy_train[0][1])
# np.save('D:/study_data/_save/_npy/keras46_5_test_x.npy',arr=xy_test[0][0])
# np.save('D:/study_data/_save/_npy/keras46_5_test_y.npy',arr=xy_test[0][1])
x_train = np.load('D:/study_data/_save/_npy/keras46_5_train_x.npy')
y_train = np.load('D:/study_data/_save/_npy/keras46_5_train_y.npy')
x_test = np.load('D:/study_data/_save/_npy/keras46_5_test_x.npy')
y_test = np.load('D:/study_data/_save/_npy/keras46_5_test_y.npy')
print(x_train,x_train.shape) #(160, 150, 150, 1)
print(y_train,y_train.shape) #(160,)
print(x_test,x_test.shape) #(120, 150, 150, 1)
print(y_test,y_test.shape) #(120,)
