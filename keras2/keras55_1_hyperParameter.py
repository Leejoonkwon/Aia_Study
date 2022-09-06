import numpy as np
# from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input,Dropout
from keras.datasets import mnist,cifar10

#1. 데이터
(x_train,y_train),(x_test,y_test) = mnist.load_data()

print(x_train.shape) # (60000, 28, 28)

x_train = x_train.reshape(60000,28*28).astype('float32')/255.
x_test = x_test.reshape(10000,28*28).astype('float32')/255.

from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

#2. 모델
def build_model(drop=0.5, optimizer='adam', activation='relu'):
    inputs = Input(shape=(28*28), name = 'input')
    x = Dense(512, activation=activation, name= 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    
    model = Model(inputs=inputs,outputs=outputs)
    
    model.compile(optimizer=optimizer,metrics=['acc'],
                  loss='sparse_categorical_crossentropy')
    return model
def create_hyperparameter():
    batchs = [100, 200 ,300, 400, 500]
    optimizers =['adam']
    dropout = [0.3, 0.4, 0.5]
    activation = ['relu','linear','sigmoid','selu','elu']
    return {"batch_size" : batchs, "optimizer":optimizers,
            "drop" : dropout, "activation": activation}
    
hyperparameters = create_hyperparameter()
print(hyperparameters)
# {'batch_size': [100, 200, 300, 400, 500],
#  'optimizer': ['adam', 'rmsprop', 'adadelta'], 
# 'dropout': [0.3, 0.4, 0.5],
# 'activation': ['relu', 'linear', 'sigmoid', 'selu', 'elu']}
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

keras_model = KerasClassifier(build_fn=build_model, verbose=1)
model = RandomizedSearchCV(keras_model,hyperparameters,cv=2, n_iter=2, refit=True) # cv,n_iter 각 최소 2번
import time
start_time = time.time()
model.fit(x_train,y_train,epochs=2,validation_split=0.4)
end_time =time.time()-start_time

print('걸린 시간 :',end_time)
print('model.best_params_ : ',model.best_params_)
print('model.best_estimators : ',model.best_estimator_)
print('model.best_score_ : ',model.best_score_)

# model.best_params_ :  
# {'optimizer': 'rmsprop', 
# 'drop': 0.4, 'batch_size': 100, 'activation': 'sigmoid'}
# model.best_estimators :  
# <tensorflow.python.keras.
# wrappers.scikit_learn.KerasClassifier object at 0x00000247E9820790>
# model.best_score_ :  0.9707333246866862
from sklearn.metrics import accuracy_score
# y_predict = model.predict(x_test)
y_predict = model.predict(x_test) 
print(y_predict)
print("accuracy_socre :",accuracy_score(y_test,y_predict))



