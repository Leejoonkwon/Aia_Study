import numpy as np      
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn import datasets

# np.save('D:/study_data/_save/_npy/keras46_5_train_x.npy',arr=xy_train[0][0])
# np.save('D:/study_data/_save/_npy/keras46_5_train_y.npy',arr=xy_train[0][1])
# np.save('D:/study_data/_save/_npy/keras46_5_test_x.npy',arr=xy_test[0][0])
# np.save('D:/study_data/_save/_npy/keras46_5_test_y.npy',arr=xy_test[0][1])
x_train = np.load('D:/study_data/_save/_npy/keras47_2_train_x.npy')
y_train = np.load('D:/study_data/_save/_npy/keras47_2_train_y.npy')
x_test = np.load('D:/study_data/_save/_npy/keras47_2_test_x.npy')
y_test = np.load('D:/study_data/_save/_npy/keras47_2_test_y.npy')


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
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일,훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
hist = model.fit(x_train,y_train,epochs=4,verbose=2,validation_split=0.25,batch_size=500)


accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss =  hist.history['loss']
val_loss =  hist.history['val_loss']

print('loss :',loss[-1])
print('val_loss :',val_loss[-1])
print('accuracy :',accuracy[-1])
print('val_accuracy :',val_accuracy[-1])

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'],marker='.',c='red',label='loss') #순차적으로 출력이므로  y값 지정 필요 x
# plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss')
# plt.grid()
# plt.title('show') #맥플러립 한글 깨짐 현상 알아서 해결해라 
# plt.ylabel('loss')
# plt.xlabel('epochs')
# # plt.legend(loc='upper right')
# plt.legend()
# plt.show()


# loss : 1.9426530599594116
# val_loss : 0.901305615901947
# accuracy : 0.5111523866653442
# val_accuracy : 0.46666666865348816