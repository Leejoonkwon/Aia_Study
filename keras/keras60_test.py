
import time
import numpy as np      
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D,Flatten,Dense,MaxPool2D,Dropout


path = 'D:\study_data\_data/' # ".은 현재 폴더"
df = pd.read_csv(path + 'music.csv'
                       )
# np.save('D:\study_data\_save\_npy\_train_x10.npy',arr=x_train)
# np.save('D:\study_data\_save\_npy\_train_y10.npy',arr=y_train)
# np.save('D:\study_data\_save\_npy\_test_x10.npy',arr=x_test)
# np.save('D:\study_data\_save\_npy\_test_y10.npy',arr=y_test)
x_train = np.load('D:\study_data\_save\_npy\_train_x10.npy')
y_train = np.load('D:\study_data\_save\_npy\_train_y10.npy')
x_test = np.load('D:\study_data\_save\_npy\_test_x10.npy')
y_test = np.load('D:\study_data\_save\_npy\_test_y10.npy')

#2. 모델 

model = Sequential()
model.add(Conv2D(input_shape=(100, 100, 1), kernel_size=(3, 3), filters=32, padding='same', activation='relu'))
model.add(Conv2D(kernel_size=(3, 3), filters=64, padding='same', activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(kernel_size=(3, 3), filters=128, padding='same', activation='relu'))
model.add(Conv2D(kernel_size=(3, 3), filters=256, padding='valid', activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(21,activation='softmax'))
model.summary()


# model.load_weights("D:\study_data\_save\keras53_project2.h5")
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
hist = model.fit(x_train,y_train,epochs=10,verbose=2,
                 validation_split=0.25,
                 callbacks=[earlyStopping])
# model.save_weights("D:\study_data\_save\keras53_project4.h5")
# model.save_weights("./_save/keras23_5_save_weights1.h5")

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
end_time = time.time()-start_time
print("걸린 시간 :",end_time)
# xy = np.load('D:\study_data\_save\_npy\_train_test.npy')
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict,axis=1)

y_test = np.argmax(y_test,axis=1)

print('y_predict :', y_predict.shape) #y_predict : (50,)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc 스코어 :', acc)




#### 증폭 후 데이터 3만개 일때
# loss : [1.3046691417694092, 0.507749080657959]
# 걸린 시간 : 168.37035512924194
# y_predict : (6775,)
# acc 스코어 : 0.5077490774907749

##########VGG16 적용 데이터 2만장일 때
# loss : [0.3138679265975952, 0.9046236276626587]
# 걸린 시간 : 92.63064312934875
# y_predict : (4477,)
# acc 스코어 : 0.9046236318963592
##########VGG16 적용 데이터 3만장일 때
# loss : [0.2217244803905487, 0.9342288374900818]
# 걸린 시간 : 129.7907133102417
# y_predict : (6477,)
# acc 스코어 : 0.934228809634089
##########VGG16 적용 데이터 4만장일 때
# loss : [0.17270071804523468, 0.9478589296340942]
# 걸린 시간 : 169.3062162399292
# y_predict : (8477,)
# acc 스코어 : 0.9478589123510676






