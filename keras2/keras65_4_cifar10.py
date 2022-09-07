# trainable = True,False 비교해보면서 만들어서 결과 비교
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.applications import vgg16
from keras.datasets import cifar10
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
from tensorflow.keras.utils import to_categorical
print(x_train.shape,y_train.shape)  # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape,y_test.shape)    # (10000, 32, 32, 3) (10000, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape,y_test.shape) # (50000, 10) (10000, 10)

#1. 데이터
# model= vgg16.VGG16() # include_top = True,input_shape=(224,224,3)이 디폴트
VGG16 = vgg16.VGG16(weights='imagenet',include_top=False,
                    input_shape=(32,32,3))

# VGG16.summary() # Trainable params: 14,714,688
# VGG16.trainable=False
# VGG16.summary() # Non-trainable params: 14,714,688
model = Sequential()
model.add(VGG16)
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(10,activation='softmax'))
# model.trainable =False
model.summary()

#3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy',optimizer='adam')

model.fit(x_train,y_train,epochs=10,batch_size=2000)
from sklearn.metrics import accuracy_score
#4. 평가, 예측
model.evaluate(x_test,y_test)
y_predcit = np.argmax(model.predict(x_test),axis=1)
y_predcit = to_categorical(y_predcit)

acc = accuracy_score(y_test,y_predcit)

print("acc : ",acc)



