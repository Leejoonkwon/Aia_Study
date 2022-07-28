import numpy as np      
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# np.save('D:\study_data\_save\_npy\_train_x4.npy',arr=x_train)
# np.save('D:\study_data\_save\_npy\_train_y4.npy',arr=y_train)
# np.save('D:\study_data\_save\_npy\_test_x4.npy',arr=x_test)
# np.save('D:\study_data\_save\_npy\_test_y4.npy',arr=y_test)
x_train = np.load('D:\study_data\_save\_npy\_train_x4.npy')
y_train = np.load('D:\study_data\_save\_npy\_train_y4.npy')
x_test = np.load('D:\study_data\_save\_npy\_test_x4.npy')
y_test = np.load('D:\study_data\_save\_npy\_test_y4.npy')
print(x_train.shape,y_train.shape)  # (34210, 150, 150, 1) (34210, 21)
print(x_test.shape,y_test.shape)    # (14662, 150, 150, 1) (14662, 21)
# print(x_test,y_test)    # (5000, 150, 150, 1) (5000, 21)

# import matplotlib.pyplot as plt
# plt.figure(figsize=(100,5))
# for i in range(300):
#     plt.subplot(6,50,i+1)
#     plt.axis('off')
#     plt.imshow(x_train[i])
# plt.show()

#2. 모델 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D,Flatten,Dense,MaxPool2D,Dropout

model = Sequential()
model.add(Conv2D(64,(3,3),input_shape=(150,150,1),padding='same',activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(21,activation='softmax'))
import time
from keras.callbacks import ModelCheckpoint,EarlyStopping

start_time = time.time()
#3. 컴파일,훈련
# filepath = './_test/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', 
                              verbose=1,restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
#                       save_best_only=True, 
#                       filepath="".join([filepath,'k24_', date, '_', filename])
#                     )
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
hist = model.fit(x_train,y_train,epochs=2,verbose=2,
                 validation_split=0.3,batch_size=5
                 ,callbacks=[earlyStopping])
model.save_weights("D:\study_data\_save\keras53_project2.h5")
# model.save_weights("./_save/keras23_5_save_weights1.h5")
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

# loss : [8.893112182617188, 0.2540000081062317]
# 걸린 시간 : 1010.0424473285675
# y_predict : (5000,)
# acc 스코어 : 0.254

# loss : [1.862046718597412, 0.24500000476837158]
# 걸린 시간 : 35.0883367061615
# y_predict : (5000,)
# acc 스코어 : 0.245







