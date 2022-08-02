from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator 
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.layers import Dense, Dropout, Input
from tensorflow.python.keras.models import Model
import numpy as np
from keras import models, layers
from keras import Input
from keras.models import Model
x_train = np.load('D:\study_data\_save\_npy\_train_x10.npy')
y_train = np.load('D:\study_data\_save\_npy\_train_y10.npy')
x_test = np.load('D:\study_data\_save\_npy\_test_x10.npy')
y_test = np.load('D:\study_data\_save\_npy\_test_y10.npy')
print(x_train.shape) #
'''
# 모델 Layer 데이터화
# input_tensor = Input(shape=(100,100,3))
pre_trained_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
pre_trained_vgg.trainable = False
pre_trained_vgg.summary()
additional_model = models.Sequential()
additional_model.add(pre_trained_vgg)
additional_model.add(layers.Flatten())
additional_model.add(layers.Dense(50, activation='relu'))
additional_model.add(layers.Dense(50, activation='relu'))
additional_model.add(layers.Dense(50, activation='relu'))
additional_model.add(layers.Dense(7, activation='softmax'))
additional_model.summary()


#3. 컴파일,훈련

additional_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
hist = additional_model.fit(x_train,y_train,epochs=10,verbose=2,
                 validation_split=0.25,
                 )

#4. 평가,예측
loss = additional_model.evaluate(x_test, y_test)
print('loss :', loss)
'''
