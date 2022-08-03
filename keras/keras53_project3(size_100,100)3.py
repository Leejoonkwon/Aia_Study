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
x_train = np.load('D:\study_data\_save\_npy\_train_x5.npy')
y_train = np.load('D:\study_data\_save\_npy\_train_y5.npy')
x_test = np.load('D:\study_data\_save\_npy\_test_x5.npy')
y_test = np.load('D:\study_data\_save\_npy\_test_y5.npy')

#2. 모델 

model = Sequential()
model.add(Conv2D(input_shape=(85, 85, 1), kernel_size=(3, 3), filters=32, padding='same', activation='relu'))
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


model.load_weights("D:\study_data\_save\keras53_project2.h5")
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
# hist = model.fit(x_train,y_train,epochs=10,verbose=2,
#                  validation_split=0.25,
#                  callbacks=[earlyStopping])
# model.save_weights("D:\study_data\_save\keras53_project2.h5")
# model.save_weights("./_save/keras23_5_save_weights1.h5")

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
end_time = time.time()-start_time
print("걸린 시간 :",end_time)
xy = np.load('D:\study_data\_save\_npy\_train_test.npy')
y_predict = model.predict(xy)
y_predict = np.argmax(y_predict,axis=1)

if y_predict < 2 :
    print('분노한 표정')
elif y_predict < 3:
    print('혐오하는 표정')
elif y_predict < 4:
    print('공포스러워하는 표정')
elif y_predict < 5:
    print('행복해하는 표정')
elif y_predict < 6:
    print('평화로운 표정')
elif y_predict < 7:
    print('슬픈 표정')
elif y_predict < 8:
    print('놀라워하는 표정')
# y_predict = np.argmax(y_predict,axis=1)
# y_test = np.argmax(y_test,axis=1)

# print('y_predict :', y_predict.shape) #y_predict : (50,)
# from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_test, y_predict)
# print('acc 스코어 :', acc)




#### 증폭 후 데이터 3만개 일때
# loss : [1.3046691417694092, 0.507749080657959]
# 걸린 시간 : 168.37035512924194
# y_predict : (6775,)
# acc 스코어 : 0.5077490774907749
##########VGG16 적용
# loss : [0.0, 1.0]
# 걸린 시간 : 121.65885210037231
# y_predict : (6000,)
# acc 스코어 : 1.0





