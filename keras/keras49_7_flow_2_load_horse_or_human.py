import numpy as np      
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn import datasets

# np.save('D:/study_data/_save/_npy/keras49_7_train_x.npy',arr=xy_df3[0][0])
# np.save('D:/study_data/_save/_npy/keras49_7_train_y.npy',arr=xy_df3[0][1])
# np.save('D:/study_data/_save/_npy/keras49_7_test_x.npy',arr=x_test)
# np.save('D:/study_data/_save/_npy/keras49_7_test_y.npy',arr=y_test)
x_train = np.load('D:/study_data/_save/_npy/keras49_7_train_x.npy')
y_train = np.load('D:/study_data/_save/_npy/keras49_7_train_y.npy')
x_test = np.load('D:/study_data/_save/_npy/keras49_7_test_x.npy')
y_test = np.load('D:/study_data/_save/_npy/keras49_7_test_y.npy')
print(y_test,y_test.shape) #(297,)

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
model.fit(x_train,y_train,epochs=250,verbose=2,validation_split=0.25,batch_size=100)


#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict,0)
print('y_predict :', y_predict.shape) #(297,1)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict) 
print('acc 스코어 :', acc)
# loss : [0.8918216228485107, 0.5096391439437866]
# y_predict : [[0.8578456 ]

#### 증폭 후
# loss : [12.333027839660645, 0.5488215684890747]
# y_predict : (297, 1)
# acc 스코어 : 0.5488215488215489
