import numpy as np
from sklearn.datasets import load_iris
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
import tensorflow as tf
tf.random.set_seed(66)
from sklearn.svm import LinearSVC #서포트벡터머신 알아서 공부해라 !!!
# DL 과 ML의 흐름은 똑같다 데이터 전처리->모델 구성 ->훈련(fit에 컴파일이 포함되어있다.) ->평가,예측
#1. 데이터
datasets = load_iris()
# print(datasets.DESCR)
# print(datasets.feature_names)
x = datasets.data
y = datasets['target']
# from sklearn.preprocessing import OneHotEncoder

# ohe = OneHotEncoder(sparse=False)
# # fit_transform은 train에만 사용하고 test에는 학습된 인코더에 fit만 해야한다
# train_cat = ohe.fit_transform(train[['cat1']])
# print(np.unique(y,return_counts=True))
# print(num)
###########(keras 버전 원핫인코딩)###############
# from tensorflow.keras.utils import to_categorical 
# y = to_categorical(y) 

# 해당 기능을 통해 y값을 클래스 수에 맞는 열로 늘리는 원핫 인코딩 처리를 한다.
#1개의 컬럼으로 [0,1,2] 였던 값을 ([1,0,0],[0,1,0],[0,0,1]과 같은 shape로 만들어줌)

###########(sklearn 버전 원핫인코딩)###############
#from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(sparse=False)
# # fit_transform은 train에만 사용하고 test에는 학습된 인코더에 fit만 해야한다
# train_cat = ohe.fit_transform(train[['cat1']])
# train_cat


# num = num.shape[0]
# print(num)

# y = np.eye(num)[data]
# print(x)
# print(y)



# print(x.shape, y.shape) #(150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25,shuffle=True ,random_state=100)
#셔플을 False 할 경우 순차적으로 스플릿하다보니 훈련에서는 나오지 않는 값이 생겨 정확도가 떨어진다.
#디폴트 값인  shuffle=True 를 통해 정확도를 올린다.
print(y_train,y_test)




#2. 모델 구성

# model = Sequential()
# model.add(Dense(100,input_dim=4))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(3, activation='softmax'))
# #다중 분류로 나오는 아웃풋 노드의 개수는 y 값의 클래스의 수와 같다.활성화함수 'softmax'를 통해 
# # 아웃풋의 합은 1이 된다.
model = LinearSVC() # DL과 다르게 단층 레이어  구성으로 연산에 걸리는 시간을 비교할 수 없다.

#3. 컴파일,훈련
# earlyStopping = EarlyStopping(monitor='loss', patience=10, mode='min', 
#                               verbose=1,restore_best_weights=True)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# hist = model.fit(x_train, y_train, epochs=1000, batch_size=40, 
#                 validation_split=0.3,
#                 callbacks = [earlyStopping],
#                 verbose=2
#                 )
#다중 분류 모델은 'categorical_crossentropy'만 사용한다 !!!!
model.fit(x_train,y_train) #####


#4.  평가,예측

results = model.score(x_test,y_test) #분류 모델과 회귀 모델에서 score를 쓰면 알아서 값이 나온다 
#ex)분류는 ACC 회귀는 R2스코어
print("results :",results)
y_predict = model.predict(x_test)
# y_test = np.argmax(y_test,axis=1)
print(y_predict)

# y_predict = np.argmax(y_predict,axis=1)
# # y_test와 y_predict의  shape가 일치해야한다.
# print(y_predict)


acc = accuracy_score(y_test, y_predict)
print('acc 스코어 :', acc)
#acc 스코어 : 0.9736842105263158

