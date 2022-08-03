

from keras.preprocessing.image import ImageDataGenerator 
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.layers import Dense, Dropout, Input
from tensorflow.python.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import numpy as np
from keras import models, layers
from keras import Input
from keras.models import Model
x_train = np.load('D:\study_data\_save\_npy\_train_x5.npy')
y_train = np.load('D:\study_data\_save\_npy\_train_y5.npy')
x_test = np.load('D:\study_data\_save\_npy\_test_x5.npy')
y_test = np.load('D:\study_data\_save\_npy\_test_y5.npy')
# print(x_train.shape) #(28709, 48, 48, 6)



######
from keras.applications.resnet import ResNet50
pre_trained_Res = ResNet50(weights='imagenet',
                           include_top=False, input_shape=(70,70,3))
pre_trained_Res.trainable = True
pre_trained_Res.summary()
additional_model = models.Sequential()
additional_model.add(pre_trained_Res)
additional_model.add(Flatten())
additional_model.add(layers.Dense(128, activation='relu'))
additional_model.add(layers.Dense(128, activation='relu'))
additional_model.add(layers.Dense(64, activation='relu'))
additional_model.add(layers.Dense(7, activation='softmax'))


#3. 컴파일,훈련
import time
start_time = time.time()
additional_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
hist = additional_model.fit(x_train,y_train,epochs=8,verbose=2,
                 validation_split=0.25,batch_size=50)
                
additional_model.save_weights("D:\study_data\_save\keras60_project10.h5")
                 
#4. 평가,예측
loss = additional_model.evaluate(x_test, y_test)
print('loss :', loss)
end_time = time.time()-start_time
print("걸린 시간 :",end_time)

y_predict = additional_model.predict(x_test)
y_predict = np.argmax(y_predict,axis=1)
y_test = np.argmax(y_test,axis=1)
# print('y_predict :',y_predict)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc 스코어 :', acc)


