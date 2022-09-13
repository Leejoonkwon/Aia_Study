# [실습] 4번 카피 복붙 
# CNN으로 딥하게 구성
# UpSampling 찾아서 이해하고 반드시 추가할 것!!!
# 3가지 개념 있다.
# - Nearest neighbor interpolation

# - Bi-linear interpolation

# - Bi-cubic interpolation

# 트랜스포즈도 찾아

import numpy as np
from keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000, 28,28,1).astype('float32')/255.

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D,MaxPooling2D,UpSampling2D,Flatten

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size, (3, 3), activation='relu', padding='same', strides=2,input_shape=(28,28,1)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
    # model.compile(optimizer='rmsprop', loss='mse')
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

model = autoencoder(hidden_layer_size=128)
# pca를 통해 0.95 이상인 n_component  몇개?
# 0.95  # 154
# 0.99  # 331
# 0.999 # 486
# 1.0   # 713
model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(x_train, x_train, epochs=10, batch_size=128,
                validation_split=0.2)
output = model.predict(x_test)


from matplotlib import pyplot as plt
import random 

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10))  = \
    plt.subplots(2, 5, figsize=(20, 7))
    
# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i ==0:
        ax.set_ylabel("INPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
  
# 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap='gray')
    if i ==0:
        ax.set_ylabel("OUTPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
plt.tight_layout()
plt.show()    