import matplotlib.pyplot  as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout 
from keras.datasets import mnist,cifar10,cifar100
import pandas as pd
import numpy as np
#1. 데이터 전처리

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
plt.imshow(x_train[3])
plt.show()