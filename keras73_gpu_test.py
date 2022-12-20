import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('CPU')
#  인자를 GPU랑 CPU를 바꿔서 실험
if gpus :
    try :
        # tf.config.experimental.set_visible_devices(gpus[0],'GPU')
        # tf.config.experimental.set_visible_devices(gpus[1],'GPU')
        tf.config.experimental.set_visible_devices(gpus[0],'CPU')
        # 두개 동시에 돌아가진 않고 1개만 택해 실행
        # gpu 환경으로 실행하게끔 하는 함수 선택적으로 실행가능하다 
        # ex) GPU1의 환경 GPU2의 환경 CPU만의 환경 
    except RuntimeError as e:
        print(e) # 예외사항 출력하기 


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
model.add(Dense(100))
model.add(Dense(10,activation='softmax'))
# model.trainable =False
model.summary()
# vgg는 안돌아감 이유 찾기
#3. 컴파일, 훈련
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=10,mode='auto')
model.compile(loss='categorical_crossentropy',optimizer='adam')

model.fit(x_train,y_train,epochs=100,batch_size=4000,
          callbacks=[es],validation_split=0.3)
from sklearn.metrics import accuracy_score
#4. 평가, 예측
model.evaluate(x_test,y_test)
y_predcit = np.argmax(model.predict(x_test),axis=1)
y_predcit = to_categorical(y_predcit)

acc = accuracy_score(y_test,y_predcit)

print("acc : ",acc)

################### include_top=False 로 진행
# acc :  0.7153
################### VGG16.trainable = False 로 진행
# acc :  0.579
################### model.trainable = False 로 진행
# acc :  0.1021


