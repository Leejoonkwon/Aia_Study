import numpy as np      
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn import datasets

# np.save('C:\_data\_save\_train_x.npy',arr=xy_df3[0][0])
# np.save('C:\_data\_save\_train_y.npy',arr=xy_df3[0][1])
# np.save('C:\_data\_save\_test_x.npy',arr=x_test)
# np.save('C:\_data\_save\_test_y.npy',arr=y_test)
x_train = np.load('C:\_data\_save\_train_x.npy')
y_train = np.load('C:\_data\_save\_train_y.npy')
x_test = np.load('C:\_data\_save\_test_x.npy')
y_test = np.load('C:\_data\_save\_test_y.npy')
print(x_train.shape,y_train.shape)  # (10000, 150, 150, 3) (10000, 7)
print(x_test.shape,y_test.shape)    # (7178, 150, 150, 3) (7178, 7)


#2. 모델 
from tensorflow.python.keras.layers import Conv2D,Flatten,Dense,MaxPool2D
from tensorflow.python.keras.models import Sequential


model = Sequential()
model.add(Conv2D(32,(2,2),input_shape=(150,150,3),padding='same',activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(16,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(7,activation='softmax'))
import time
start_time = time.time()
from keras.callbacks import ModelCheckpoint,EarlyStopping
earlyStopping = EarlyStopping(monitor='loss', patience=10, mode='min', 
                              verbose=1,restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
#                       save_best_only=True, 
#                       filepath="".join([filepath,'k24_', date, '_', filename])
#                     )
#3. 컴파일,훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
hist = model.fit(x_train,y_train,epochs=250,verbose=2,
                 validation_split=0.25,batch_size=50,
                 callbacks=[earlyStopping])


#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
end_time = time.time()-start_time
print("걸린 시간 :",end_time)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict,axis=1)
y_test = np.argmax(y_test,axis=1)

# print('y_predict :', y_predict.shape) #y_predict : (50,)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc 스코어 :', acc)
# loss : 1.9426530599594116
# val_loss : 0.901305615901947

#### 증폭 후
# loss : [539.5560302734375, 0.23999999463558197]
# y_predict : (50,)
# acc 스코어 : 0.24
