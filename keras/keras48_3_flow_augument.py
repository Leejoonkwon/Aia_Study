from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator 
import numpy as np
#1.데이터
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()   

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)
# print(x_train.shape[0]) #60000
# print(x_train.shape[1]) #28
# print(x_train.shape[2]) #28


augument_size = 40000
randidx = np.random.randint(x_train.shape[0],size=augument_size)
print(randidx,randidx.shape) #(40000,)
print(np.min(randidx),np.max(randidx))  #0 59997
print(type(randidx)) #<class 'numpy.ndarray'>

x_augumented = x_train[randidx].copy()
y_augumented = y_train[randidx].copy()

# print(x_augumented.shape)
# print(y_augumented.shape)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)

x_augumented = x_augumented.reshape(x_augumented.shape[0],
                                    x_augumented.shape[1],
                                    x_augumented.shape[2], 1)

x_augumented = train_datagen.flow(x_augumented,y_augumented,
                                  batch_size=augument_size,shuffle=False).next()[0]
    
print(x_augumented,x_augumented.shape) #(40000, 28, 28, 1)
x_train = np.concatenate((x_train,x_augumented)) # 소괄호() 1개와  소괄호(()) 2개의 차이를 공부해라!
y_train = np.concatenate((y_train,y_augumented)) 
# 소괄호() 1개와  소괄호(()) 2개의 차이를 공부해라! 2개인 이유는 안에 옵션을 더 넣을 수 있기 때문이다.아무 것도 안하면 디폴트로 들어감
print(x_train.shape,y_train.shape) #(100000, 28, 28, 1) (100000,)

#2. 모델 구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten

model = Sequential()
model.add(Conv2D(64,(2,2),input_shape=(28,28,1),padding='same',activation='relu'))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='relu'))

#3. 컴파일,훈련
model.compile(loss='mae',optimizer='adam')
model.fit(x_train,y_train,epochs=100,verbose=2,batch_size=1024,validation_split=0.25)
# model.fit_generator(x_train,y_train,epochs=3,
#                     validation_data=[x_train,y_train],
#                     )

#4. 평가,예측
loss = model.evaluate(x_test,y_test)
print("loss :",loss)
y_predict = model.predict(x_test)
print(y_predict)

