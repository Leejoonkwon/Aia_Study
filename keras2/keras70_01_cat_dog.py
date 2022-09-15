#전이학습으로 맹그로~
import numpy as np      
from tensorflow.python.keras.callbacks import EarlyStopping
x_train = np.load('D:/study_data/_save/_npy/keras49_6_train_x.npy')
y_train = np.load('D:/study_data/_save/_npy/keras49_6_train_y.npy')
x_test = np.load('D:/study_data/_save/_npy/keras49_6_test_x.npy')
y_test = np.load('D:/study_data/_save/_npy/keras49_6_test_y.npy')
# print(x_train.shape, y_train.shape)    #  (5000, 150, 150, 3) (5000,)
# print(x_test.shape, y_test.shape)      #  (2023, 150, 150, 3) (2023,)
from keras.applications import vgg16,vgg19
from keras.applications import resnet,resnet_v2
from keras.applications import resnet
from keras.applications import densenet
from keras.applications import inception_v3,inception_resnet_v2
from keras.applications import mobilenet,mobilenet_v2,mobilenet_v3
from keras.applications import nasnet
from keras.applications import efficientnet
from keras.applications import xception
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,GlobalAveragePooling2D
pretrain = [#vgg16.VGG16,vgg19.VGG19,resnet.ResNet50,
            # resnet.ResNet101,resnet.ResNet152,
            # resnet_v2.ResNet50V2,resnet_v2.ResNet101V2,resnet_v2.ResNet152V2,
            # densenet.DenseNet121,densenet.DenseNet169,densenet.DenseNet201,
            # mobilenet.MobileNet,mobilenet_v2.MobileNetV2,mobilenet_v3.MobileNetV3Small,
            # mobilenet_v3.MobileNetV3Large,
            efficientnet.EfficientNetB0,efficientnet.EfficientNetB1,
            efficientnet.EfficientNetB7,xception.Xception]

#2. 모델 

for i in pretrain:
    pretr = i(include_top=False,weights='imagenet',input_shape=(150,150,3)) 
    model = Sequential()
    model.add(pretr)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))

    #3. 컴파일,훈련
    es = EarlyStopping(monitor='val_loss',patience=5,mode='auto')
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    hist = model.fit(x_train,y_train,epochs=10,verbose=0,
                    validation_split=0.25,batch_size=10
                    ,callbacks=[es])

    #4. 평가,예측
    loss = model.evaluate(x_test, y_test)
    print('loss :', loss)
    y_predict = model.predict(x_test)
    y_predict = np.round(y_predict,0)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_predict) 
    print('model : ',i.__name__,'\t','Acc : ', acc)

# loss : [0.8918216228485107, 0.5096391439437866]
# y_predict : [[0.8578456 ]

# model :  VGG16   Acc :  0.5002471576866041
# model :  ResNet50        Acc :  0.49975284231339595
# model :  ResNet101       Acc :  0.4957983193277311
# model :  ResNet152       Acc :  0.5007414730598122
# model :  ResNet50V2      Acc :  0.49925852694018785
# model :  ResNet101V2     Acc :  0.49975284231339595
# model :  ResNet152V2     Acc :  0.49975284231339595
# model :  DenseNet121     Acc :  0.5002471576866041
# model :  DenseNet169     Acc :  0.5042016806722689
# model :  DenseNet201     Acc :  0.49975284231339595
# model :  MobileNet       Acc :  0.5333662876915473
# model :  MobileNetV2     Acc :  0.46465645081562035
# model :  MobileNetV3Small        Acc :  0.5002471576866041
# model :  MobileNetV3Large        Acc :  0.5002471576866041