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
target_size = 90
xy_train= test_datagen.flow_from_directory(
    'D:\study_data\project\\train',
    target_size=(target_size,target_size),
    class_mode='categorical',
    # color_mode='grayscale',
    batch_size=28709,
    shuffle=True,) # 28709
xy_test = test_datagen.flow_from_directory(
    'D:\study_data\project\\test',
    target_size=(target_size,target_size),
    class_mode='categorical',
    batch_size=10000,
    # color_mode='grayscale',
    shuffle=False) # 7178
# xy_test2 = test_datagen.flow_from_directory(
#     'D:\study_data\\test2',
#     target_size=(target_size,target_size),
#     class_mode='categorical',
#     batch_size=10000,
#     # color_mode='grayscale',
#     shuffle=False) # 7178
# np.save('D:\study_data\_save\_npy\_train_x18.npy',arr=xy_test2[0][0])
# x = xy_train[0][0]
# y = xy_train[0][1]
# x1 = x[:435]
# y1 = y[:435]
# augument_size = 3000
# randidx = np.random.randint(x1.shape[0],size=augument_size)

# x_augumented = x[randidx].copy()
# y_augumented = y[randidx].copy()

# x_augumented = train_datagen.flow(x_augumented,y_augumented,
#                                   batch_size=augument_size,shuffle=False)
    
# # print(x_augumented,x_augumented.shape) #(40000, 28, 28, 1)
# x_train = np.concatenate((x,x_augumented[0][0])) # 소괄호() 1개와  소괄호(()) 2개의 차이를 공부해라!
# y_train = np.concatenate((y,y_augumented)) 



np.save('D:\study_data\_save\_npy\_train_x5.npy',arr=xy_train[0][0])
np.save('D:\study_data\_save\_npy\_train_y5.npy',arr=xy_train[0][1])
np.save('D:\study_data\_save\_npy\_test_x5.npy',arr=xy_test[0][0])
np.save('D:\study_data\_save\_npy\_test_y5.npy',arr=xy_test[0][1])






