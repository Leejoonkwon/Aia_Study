import numpy as np      
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn import datasets

# np.save('D:/study_data/_save/_npy/keras49_1_train_x.npy',arr=xy_df3[0][0])
# np.save('D:/study_data/_save/_npy/keras49_1_train_y.npy',arr=xy_df3[0][1])
# np.save('D:/study_data/_save/_npy/keras49_1_test_x.npy',arr=x_test)
# np.save('D:/study_data/_save/_npy/keras49_1_test_y.npy',arr=y_test)
x_train = np.load('D:/study_data/_save/_npy/keras49_5_train_x.npy')
y_train = np.load('D:/study_data/_save/_npy/keras49_5_train_y.npy')
x_test = np.load('D:/study_data/_save/_npy/keras49_5_test_x.npy')
y_test = np.load('D:/study_data/_save/_npy/keras49_5_test_y.npy')
print(x_train,x_train.shape) #(160, 150, 150, 1)
print(y_train,y_train.shape) #(160,)
print(x_test,x_test.shape) #(120, 150, 150, 1)
print(y_test,y_test.shape) #(120,)

#2. 모델 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D,Flatten,Dense,MaxPool2D

model = Sequential()
model.add(Conv2D(32,(2,2),input_shape=(150,150,1),padding='same',activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일,훈련
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=30,verbose=2,validation_split=0.2,batch_size=50)# 배치를 최대로 잡을 경우 가능한 구조

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
print('y_predict :', y_predict) 
from sklearn.metrics import accuracy_score
y_predict = np.round(y_predict,0)
acc = accuracy_score(y_test, y_predict) 
print('acc 스코어 :', acc)
# loss : [0.9344231486320496, 0.5]
# y_predict : [0.1711559]