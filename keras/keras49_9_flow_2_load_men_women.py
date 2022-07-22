import numpy as np      
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn import datasets

# np.save('D:/study_data/_save/_npy/keras46_5_train_x.npy',arr=xy_train[0][0])
# np.save('D:/study_data/_save/_npy/keras46_5_train_y.npy',arr=xy_train[0][1])
# np.save('D:/study_data/_save/_npy/keras46_5_test_x.npy',arr=xy_test[0][0])
# np.save('D:/study_data/_save/_npy/keras46_5_test_y.npy',arr=xy_test[0][1])
x_train = np.load('D:/study_data/_save/_npy/keras49_9_train_x.npy')
y_train = np.load('D:/study_data/_save/_npy/keras49_9_train_y.npy')
x_test = np.load('D:/study_data/_save/_npy/keras49_9_test_x.npy')
y_test = np.load('D:/study_data/_save/_npy/keras49_9_test_y.npy')
# z_test = np.load('D:/study_data/_save/_npy/keras47_4_test_z.npy')


#2. 모델 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D,Flatten,Dense,MaxPool2D

model = Sequential()
model.add(Conv2D(32,(2,2),input_shape=(150,150,3),padding='same',activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64,(2,2),activation='relu'))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일,훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
hist = model.fit(x_train,y_train,epochs=50,verbose=2,validation_split=0.25,batch_size=50)


#4. 평가,훈련
loss = model.evaluate(x_test, y_test)
print("loss :",loss)
print("====================")


y_predict = model.predict(x_test)
# 증폭 후
# loss : [11.444925308227539, 0.5600000023841858]
if 	y_predict[-1] >= 0.5 :
    print('여자다') # 출력값: 
else :
    print('남자다') # 출력값:
#여자다
