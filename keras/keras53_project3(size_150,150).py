from keras.preprocessing.image import ImageDataGenerator 
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.layers import Dense, Dropout, Input,MaxPool2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.keras import layers, models
img_input = layers.Input(shape=(100,100,1))

def conv_block(inputs, filter=64, kernel_size=(3, 3), stride=(1, 1), padding='same', activation='relu', block=1, layer=1):
	x = layers.Conv2D(filters=filter, kernel_size=kernel_size, strides=stride, padding=padding, activation=activation , name=f'block{block}_conv{layer}')(inputs)
	return x

def VGG16():
  # Block 1
  x = conv_block(img_input, filter=64, block=1, layer=1)
  x = conv_block(x, filter=64, block=1, layer=2)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

  # Block 2
  x = conv_block(x, filter=128, block=2, layer=1)
  x = conv_block(x, filter=128, block=2, layer=2)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

  # Block 3
  x = conv_block(x, filter=256, block=3, layer=1)
  x = conv_block(x, filter=256, block=3, layer=2)
  x = conv_block(x, filter=256, block=3, layer=3)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

  # Block 4
  x = conv_block(x, filter=512, block=4, layer=1)
  x = conv_block(x, filter=512, block=4, layer=2)
  x = conv_block(x, filter=512, block=4, layer=3)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

  # Block 5
  x = conv_block(x, filter=512, block=5, layer=1)
  x = conv_block(x, filter=512, block=5, layer=2)
  x = conv_block(x, filter=512, block=5, layer=3)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

  # Classification block
  x = layers.Flatten(name='flatten')(x)
  x = layers.Dense(4096, activation='relu', name='fc1')(x)
  x = layers.Dense(4096, activation='relu', name='fc2')(x)
  x = layers.Dense(1000, activation='softmax', name='predictions')(x)

  # Create model.
  model = models.Model(inputs, x, name='vgg16')
  return model
  
vgg16_model = VGG16()




