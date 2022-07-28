import numpy as np      
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16

# np.save('D:\study_data\_save\_npy\_train_x5.npy',arr=x_train)
# np.save('D:\study_data\_save\_npy\_train_y5.npy',arr=y_train)
# np.save('D:\study_data\_save\_npy\_test_x5.npy',arr=x_test)
# np.save('D:\study_data\_save\_npy\_test_y5.npy',arr=y_test)
###########size 100,100으로 한 파일
 
x_train = np.load('D:\study_data\_save\_npy\_train_x5.npy')
y_train = np.load('D:\study_data\_save\_npy\_train_y5.npy')
x_test = np.load('D:\study_data\_save\_npy\_test_x5.npy')
y_test = np.load('D:\study_data\_save\_npy\_test_y5.npy')
# print(x_train.shape,y_train.shape)  # (27097, 100, 100, 1) (27097, 21)
# print(x_test.shape,y_test.shape) # (6775, 100, 100, 1) (6775, 21)

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
# conv_base = VGG16(weights='imagenet',
#                   include_top=False,
#                   input_shape=(100,100,3))
model = Sequential()
model.add(Conv2D(128,(2,2),input_shape=(100,100,1),padding='same',activation='relu'))
# model.add(conv_base)
model.add(MaxPool2D((2,2)))
model.add(Conv2D(128,(2,2),padding='same',activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(128,(2,2),padding='same',activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(21,activation='softmax'))
model.summary()

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
hist = model.fit(x_train,y_train,epochs=150,verbose=2,
                 validation_split=0.3,
                 batch_size=500,
                 callbacks=[earlyStopping])
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
# loss : [1.5243678092956543, 0.39542436599731445]
# 걸린 시간 : 47.57860064506531
# y_predict : (6775,)
# acc 스코어 : 0.39542435424354244
######
# loss : [1.5229676961898804, 0.39261993765830994]
# 걸린 시간 : 48.090755462646484
# y_predict : (6775,)
# acc 스코어 : 0.392619926199262
######
# loss : [1.552404522895813, 0.3648708462715149]
# 걸린 시간 : 21.46925687789917
# y_predict : (6775,)
# acc 스코어 : 0.36487084870848707
######
# 걸린 시간 : 137.71111226081848
# y_predict : (6775,)
# acc 스코어 : 0.4277490774907749






