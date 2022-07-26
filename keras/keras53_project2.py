import numpy as np      
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn import datasets

# np.save('D:\study_data\_save\_npy\_train_x1.npy',arr=xy_df3[0][0])
# np.save('D:\study_data\_data\_save\_npy\_train_y1.npy',arr=xy_df3[0][1])
# np.save('D:\study_data\_data\_save\_npy\_test_x1.npy',arr=x_test)
# np.save('D:\study_data\_data\_save\_npy\_test_y1.npy',arr=y_test)
x_train = np.load('D:\study_data\_save\_npy\_train_x1.npy')
y_train = np.load('D:\study_data\_save\_npy\_train_y1.npy')
x_test = np.load('D:\study_data\_save\_npy\_test_x1.npy')
y_test = np.load('D:\study_data\_save\_npy\_test_y1.npy')
print(x_train.shape,y_train.shape)  # (15000, 150, 150, 1) (15000, 21)
print(x_test.shape,y_test.shape)    # (5000, 150, 150, 1) (5000, 21)


#2. 모델 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D,Flatten,Dense,MaxPool2D

model = Sequential()
model.add(Conv2D(64,(2,2),input_shape=(150,150,1),padding='same',activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(21,activation='softmax'))
import time
start_time = time.time()
#3. 컴파일,훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
hist = model.fit(x_train,y_train,epochs=2,verbose=2,validation_split=0.25,batch_size=30)
# model.save_weights

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
end_time = time.time()-start_time
print("걸린 시간 :",end_time)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict,axis=1)
y_test = np.argmax(y_test,axis=1)

print('y_predict :', y_predict.shape) #y_predict : (50,)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc 스코어 :', acc)
#### 증폭 후
#acc 스코어 : 0.1812









