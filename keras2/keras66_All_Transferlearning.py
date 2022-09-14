import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten,GlobalAveragePooling2D
from keras.datasets import cifar100
from keras.applications import VGG16,VGG19
from keras.applications import ResNet50,ResNet50V2
from keras.applications import ResNet101,ResNet101V2,ResNet152,ResNet152V2
from keras.applications import DenseNet121,DenseNet169,DenseNet201
from keras.applications import InceptionV3,InceptionResNetV2
from keras.applications import MobileNet,MobileNetV2
from keras.applications import MobileNetV3Small,MobileNetV3Large
from keras.applications import NASNetLarge, NASNetMobile
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from keras.applications import Xception

pretrain = [VGG16,VGG19,ResNet50,ResNet50V2,ResNet101,ResNet101V2,ResNet152,
            ResNet152V2,DenseNet121,DenseNet169,DenseNet201,InceptionV3,InceptionResNetV2,
            MobileNet,MobileNetV2,MobileNetV3Small,MobileNetV3Large,NASNetLarge,NASNetMobile,
            EfficientNetB0,EfficientNetB1,EfficientNetB7,Xception]
# (x_train,y_train),(x_test,y_test) = cifar100.load_data()
# from tensorflow.keras.utils import to_categorical
# print(x_train.shape,y_train.shape)  # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape,y_test.shape)    # (10000, 32, 32, 3) (10000, 1)
# # y_train = to_categorical(y_train)
# # y_test = to_categorical(y_test)
# print(y_train.shape,y_test.shape) # (50000, 100) (10000, 100)


#1. 데이터
# model= vgg16.VGG16() # include_top = True,input_shape=(224,224,3)이 디폴트
for i in pretrain:
    pre = i(weights='imagenet')
    # VGG16.summary() # Trainable params: 14,714,688
    # VGG16.trainable=False
    # VGG16.summary() # Non-trainable params: 14,714,688
    # model = Sequential()
    # model.add(pre)
    # model.add(GlobalAveragePooling2D())
    # model.add(Dense(100))
    # model.add(Dense(10,activation='softmax'))
    # model.trainable =False
    # model.summary()
    #3. 컴파일, 훈련
    # from tensorflow.python.keras.callbacks import EarlyStopping
    # es = EarlyStopping(monitor='val_loss',patience=10,mode='auto')
    # model.compile(loss='sparse_categorical_crossentropy',optimizer='adam')
    # model.fit(x_train,y_train,epochs=10,batch_size=4000,
    #         callbacks=[es],validation_split=0.3,verbose=1)
    # from sklearn.metrics import accuracy_score
    # #4. 평가, 예측
    # model.evaluate(x_test,y_test)
    # y_predcit = np.argmax(model.predict(x_test),axis=1)
    # acc = accuracy_score(y_test,y_predcit)
    print("모델명 : ",i.__name__)
    print("전체 가중치 갯수 : ",len(pre.weights),"\t 전체 훈련 가능 가중치 갯수 : ",len(pre.trainable_weights))
    print("----Trainable = False 일 경우-------")
    pre.trainable=False
    print("전체 가중치 갯수 : ",len(pre.weights),"\t 전체 훈련 가능 가중치 갯수 : ",len(pre.trainable_weights))
    print("==================================")



# 모델명 :  VGG16
# 전체 가중치 갯수 :  32   전체 훈련 가능 가중치 갯수 :  32
# ----Trainable = False 일 경우-------
# 전체 가중치 갯수 :  32  전체 훈련 가능 가중치 갯수 :  0
# ==================================
# 모델명 :  VGG19
# 전체 가중치 갯수 :  38   전체 훈련 가능 가중치 갯수 :  38
# ----Trainable = False 일 경우-------
# 전체 가중치 갯수 :  38  전체 훈련 가능 가중치 갯수 :  0
# ==================================
# 모델명 :  ResNet50
# 전체 가중치 갯수 :  320          전체 훈련 가능 가중치 갯수 :  214
# ----Trainable = False 일 경우-------
# 전체 가중치 갯수 :  320         전체 훈련 가능 가중치 갯수 :  0
# ==================================
# 모델명 :  ResNet50V2
# 전체 가중치 갯수 :  272          전체 훈련 가능 가중치 갯수 :  174
# ----Trainable = False 일 경우-------
# 전체 가중치 갯수 :  272         전체 훈련 가능 가중치 갯수 :  0
# ==================================
# 모델명 :  ResNet101
# 전체 가중치 갯수 :  626          전체 훈련 가능 가중치 갯수 :  418
# ----Trainable = False 일 경우-------
# 전체 가중치 갯수 :  626         전체 훈련 가능 가중치 갯수 :  0
# ==================================
# 모델명 :  ResNet101V2
# 전체 가중치 갯수 :  544          전체 훈련 가능 가중치 갯수 :  344
# ----Trainable = False 일 경우-------
# 전체 가중치 갯수 :  544         전체 훈련 가능 가중치 갯수 :  0
# ==================================
# 모델명 :  ResNet152
# 전체 가중치 갯수 :  932          전체 훈련 가능 가중치 갯수 :  622
# ----Trainable = False 일 경우-------
# 전체 가중치 갯수 :  932         전체 훈련 가능 가중치 갯수 :  0
# ==================================
# 모델명 :  ResNet152V2
# 전체 가중치 갯수 :  816          전체 훈련 가능 가중치 갯수 :  514
# ----Trainable = False 일 경우-------
# 전체 가중치 갯수 :  816         전체 훈련 가능 가중치 갯수 :  0
# ==================================
# 모델명 :  DenseNet121
# 전체 가중치 갯수 :  606          전체 훈련 가능 가중치 갯수 :  364
# ----Trainable = False 일 경우-------
# 전체 가중치 갯수 :  606         전체 훈련 가능 가중치 갯수 :  0
# ==================================
# 모델명 :  DenseNet169
# 전체 가중치 갯수 :  846          전체 훈련 가능 가중치 갯수 :  508
# ----Trainable = False 일 경우-------
# 전체 가중치 갯수 :  846         전체 훈련 가능 가중치 갯수 :  0
# ==================================
# 모델명 :  DenseNet201
# 전체 가중치 갯수 :  1006         전체 훈련 가능 가중치 갯수 :  604
# ----Trainable = False 일 경우-------
# 전체 가중치 갯수 :  1006        전체 훈련 가능 가중치 갯수 :  0
# ==================================
# 모델명 :  InceptionV3
# 전체 가중치 갯수 :  378          전체 훈련 가능 가중치 갯수 :  190
# ----Trainable = False 일 경우-------
# 전체 가중치 갯수 :  378         전체 훈련 가능 가중치 갯수 :  0
# ==================================
# 모델명 :  InceptionResNetV2
# 전체 가중치 갯수 :  898          전체 훈련 가능 가중치 갯수 :  490
# ----Trainable = False 일 경우-------
# 전체 가중치 갯수 :  898         전체 훈련 가능 가중치 갯수 :  0
# ==================================
# 모델명 :  MobileNet
# 전체 가중치 갯수 :  137          전체 훈련 가능 가중치 갯수 :  83
# ----Trainable = False 일 경우-------
# 전체 가중치 갯수 :  137         전체 훈련 가능 가중치 갯수 :  0
# ==================================
# 모델명 :  MobileNetV2
# 전체 가중치 갯수 :  262          전체 훈련 가능 가중치 갯수 :  158
# ----Trainable = False 일 경우-------
# 전체 가중치 갯수 :  262         전체 훈련 가능 가중치 갯수 :  0
# ==================================
# WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not 224. Weights for input shape (224, 224) will be loaded as the default.
# 모델명 :  MobileNetV3Small
# 전체 가중치 갯수 :  210          전체 훈련 가능 가중치 갯수 :  142
# ----Trainable = False 일 경우-------
# 전체 가중치 갯수 :  210         전체 훈련 가능 가중치 갯수 :  0
# ==================================
# WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not 224. Weights for input shape (224, 224) will be loaded as the default.
# 모델명 :  MobileNetV3Large
# 전체 가중치 갯수 :  266          전체 훈련 가능 가중치 갯수 :  174
# ----Trainable = False 일 경우-------
# 전체 가중치 갯수 :  266         전체 훈련 가능 가중치 갯수 :  0
# ==================================
# 모델명 :  NASNetLarge
# 전체 가중치 갯수 :  1546         전체 훈련 가능 가중치 갯수 :  1018
# ----Trainable = False 일 경우-------
# 전체 가중치 갯수 :  1546        전체 훈련 가능 가중치 갯수 :  0
# ==================================
# 모델명 :  NASNetMobile
# 전체 가중치 갯수 :  1126         전체 훈련 가능 가중치 갯수 :  742
# ----Trainable = False 일 경우-------
# 전체 가중치 갯수 :  1126        전체 훈련 가능 가중치 갯수 :  0
# ==================================
# 모델명 :  EfficientNetB0
# 전체 가중치 갯수 :  314          전체 훈련 가능 가중치 갯수 :  213
# ----Trainable = False 일 경우-------
# 전체 가중치 갯수 :  314         전체 훈련 가능 가중치 갯수 :  0
# ==================================
# 모델명 :  EfficientNetB1
# 전체 가중치 갯수 :  442          전체 훈련 가능 가중치 갯수 :  301
# ----Trainable = False 일 경우-------
# 전체 가중치 갯수 :  442         전체 훈련 가능 가중치 갯수 :  0
# ==================================
# 모델명 :  EfficientNetB7
# 전체 가중치 갯수 :  1040         전체 훈련 가능 가중치 갯수 :  711
# ----Trainable = False 일 경우-------
# 전체 가중치 갯수 :  1040        전체 훈련 가능 가중치 갯수 :  0
# ==================================
# 모델명 :  Xception
# 전체 가중치 갯수 :  236          전체 훈련 가능 가중치 갯수 :  156
# ----Trainable = False 일 경우-------
# 전체 가중치 갯수 :  236         전체 훈련 가능 가중치 갯수 :  0
# ==================================


