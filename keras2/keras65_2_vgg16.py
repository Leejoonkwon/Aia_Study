import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.applications import vgg16,resnet_v2




#1. 데이터
# model= vgg16.VGG16() # include_top = True,input_shape=(224,224,3)이 디폴트
VGG16 = vgg16.VGG16(weights='imagenet',include_top=False,
                    input_shape=(32,32,3))

# VGG16.summary() # Trainable params: 14,714,688
# VGG16.trainable=False
# VGG16.summary() # Non-trainable params: 14,714,688
resnet_v2.ResNet152V2
model = Sequential()
model.add(VGG16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10))
# model.trainable =False
model.summary()  

                                    #Trainable: True  | VGG16 False  | # model False
print(len(model.weights))                        # 30           # 30           # 30
print(len(model.trainable_weights))              # 30           # 4            # 0

############################## include_top = False 후 커스터마이징 시
# 1. Trainable을 디폴트인 True할 경우 weights를 레이어를 통과할 때마다 갱신한다.
# 2. VGG16 중 FC  이후 레이어만 weights를  갱신한다.
# 3. 모든 layer에서 weights를 갱신하지 않는다. 가중치 동결!!


