

from keras.preprocessing.image import ImageDataGenerator 
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.layers import Dense, Dropout, Input
from tensorflow.python.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import numpy as np
from keras import models, layers
from keras import Input
from keras.models import Model
x_train = np.load('D:\study_data\_save\_npy\_train_x11.npy')
y_train = np.load('D:\study_data\_save\_npy\_train_y11.npy')
x_test = np.load('D:\study_data\_save\_npy\_test_x11.npy')
y_test = np.load('D:\study_data\_save\_npy\_test_y11.npy')
# print(x_train.shape) #(28709, 48, 48, 6)

# 모델 


from keras.applications.resnet import ResNet50
pre_trained_Res = ResNet50(weights='imagenet', include_top=False, input_shape=(48,48,3),classes=7)
pre_trained_Res.summary()

additional_model = models.Sequential()
additional_model.add(pre_trained_Res)
additional_model.add(GlobalAveragePooling2D())
additional_model.add(layers.Dense(128, activation='relu'))
additional_model.add(layers.Dense(128, activation='relu'))
additional_model.add(layers.Dense(64, activation='relu'))
additional_model.add(layers.Dense(7, activation='softmax'))

# additional_model.summary()


#3. 컴파일,훈련
import time
start_time = time.time()
additional_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
hist = additional_model.fit(x_train,y_train,epochs=20,verbose=2,
                 validation_split=0.25,batch_size=100)
                
                 
#4. 평가,예측
loss = additional_model.evaluate(x_test, y_test)
print('loss :', loss)
end_time = time.time()-start_time
y_predict = additional_model.predict(x_test)
y_predict = np.argmax(y_predict,axis=1)
y_test = np.argmax(y_test,axis=1)
print('y_predict :',y_predict)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc 스코어 :', acc)
print("걸린 시간 :",end_time)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'],marker='.',c='red',label='loss') #순차적으로 출력이므로  y값 지정 필요 x
plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss')
plt.grid()
plt.title('평가표') #맥플러립 한글 깨짐 현상 알아서 해결해라 
plt.ylabel('loss')
plt.xlabel('epochs')
# plt.legend(loc='upper right')
plt.legend()
plt.show()

