import numpy as np      
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn import datasets

# np.save('D:/study_data/_save/_npy/keras49_1_train_x.npy',arr=x_train)
# np.save('D:/study_data/_save/_npy/keras49_1_train_y.npy',arr=y_train)
# np.save('D:/study_data/_save/_npy/keras49_1_test_x.npy',arr=x_test)
# np.save('D:/study_data/_save/_npy/keras49_1_test_y.npy',arr=y_test)
x_train = np.load('D:/study_data/_save/_npy/keras49_1_train_x.npy')
y_train = np.load('D:/study_data/_save/_npy/keras49_1_train_y.npy')
x_test = np.load('D:/study_data/_save/_npy/keras49_1_test_x.npy')
y_test = np.load('D:/study_data/_save/_npy/keras49_1_test_y.npy')


#2. 모델 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D,Flatten,Dense,MaxPool2D

model = Sequential()
model.add(Conv2D(32,(2,2),input_shape=(28,28,1),padding='same',activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일,훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=4,verbose=2,validation_split=0.25,batch_size=500)
# hist = model.fit_generator(x_train,y_train,epochs=2,
#                     validation_split=0.25,
#                     steps_per_epoch=32,
#                     validation_steps=4) # 배치가 최대 아닐 경우 사용

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score,accuracy_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)
#증폭 후 
# loss : [-2273.6396484375, 0.11900000274181366]
# r2스코어 : -1.4131460288439657

