import numpy as np      
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn import datasets

# np.save('D:/study_data/_save/_npy/keras46_5_train_x.npy',arr=xy_train[0][0])
# np.save('D:/study_data/_save/_npy/keras46_5_train_y.npy',arr=xy_train[0][1])
# np.save('D:/study_data/_save/_npy/keras46_5_test_x.npy',arr=xy_test[0][0])
# np.save('D:/study_data/_save/_npy/keras46_5_test_y.npy',arr=xy_test[0][1])
x_train = np.load('D:/study_data/_save/_npy/keras49_8_train_x.npy')
y_train = np.load('D:/study_data/_save/_npy/keras49_8_train_y.npy')
x_test = np.load('D:/study_data/_save/_npy/keras49_8_test_x.npy')
y_test = np.load('D:/study_data/_save/_npy/keras49_8_test_y.npy')


#2. 모델 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D,Flatten,Dense,MaxPool2D

model = Sequential()
model.add(Conv2D(32,(2,2),input_shape=(150,150,3),padding='same',activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(3,activation='softmax'))

#3. 컴파일,훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
hist = model.fit(x_train,y_train,epochs=4,verbose=2,validation_split=0.25,batch_size=50)


#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
print('y_predict :', y_predict) #(10000, 100)
# loss : 1.9426530599594116
# val_loss : 0.901305615901947

#### 증폭 후
# loss : [24.979812622070312, 0.4399999976158142]
# y_predict : [[0.99974626]
