import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import pandas as pd 
#1.데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(581012, 54) (581012,)

# from sklearn.preprocessing import OneHotEncoder
print('y의 라벨값 :', np.unique(y,return_counts=True))
# y의 라벨값 : (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
    #   dtype=int64))


###########(sklearn 버전 원핫인코딩)###############
from sklearn.preprocessing import OneHotEncoder
y = np.array(y).reshape(-1,1)
print(y) # (581012, 8)

ohe = OneHotEncoder()
ohe.fit(y)

y_class = ohe.transform(y).toarray()

print(y_class.shape) # (581012, 7)

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



# print(x.shape, y.shape) #(581012, 54) (581012,)

x_train, x_test, y_train, y_test = train_test_split(x,y_class, test_size=0.15,shuffle=True ,random_state=100)
#셔플을 False 할 경우 순차적으로 스플릿하다보니 훈련에서는 나오지 않는 값이 생겨 정확도가 떨어진다.
#디폴트 값인  shuffle=True 를 통해 정확도를 올린다.
print(y_train,y_test)




#2. 모델 구성

model = Sequential()
model.add(Dense(100,input_dim=54))
model.add(Dense(100, activation='relu'))
model.add(Dense(7, activation='softmax'))
#다중 분류로 나오는 아웃풋 노드의 개수는 y 값의 클래스의 수와 같다.활성화함수 'softmax'를 통해 
# 아웃풋의 합은 1이 된다.


#3. 컴파일,훈련
earlyStopping = EarlyStopping(monitor='loss', patience=150, mode='min', 
                              verbose=1,restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=15000, batch_size=14000, 
                validation_split=0.3,
                callbacks = [earlyStopping],
                verbose=2
                )
#다중 분류 모델은 'categorical_crossentropy'만 사용한다 !!!!단 이진 분류 모델에도 카테고리 크로스 엔트로피를 사용 가능하지만
#다중 분류 모델에서 이진 분류인 바이너리 크로스 엔트로피는 사용 불가하다.

#4.  평가,예측

# loss,acc = model.evaluate(x_test,y_test)
# print('loss :',loss)
# print('accuracy :',acc)
# print("+++++++++  y_test       +++++++++")
# print(y_test[:5])
# print("+++++++++  y_pred     +++++++++++++")
# result = model.evaluate(x_test,y_test) 위에와 같은 개념 [0] 또는 [1]을 통해 출력가능
# print('loss :',result[0])
# print('accuracy :',result[1])

y_predict = model.predict(x_test)
y_test = np.argmax(y_test,axis=1)
print(y_test)

y_predict = np.argmax(y_predict,axis=1)
# y_test와 y_predict의  shape가 일치해야한다.
print(y_predict)


acc = accuracy_score(y_test, y_predict)
print('acc 스코어 :', acc)

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'],marker='.',c='red',label='loss') #순차적으로 출력이므로  y값 지정 필요 x
# plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss')
# plt.grid()
# plt.title('영어싫어') #맥플러립 한글 깨짐 현상 알아서 해결해라 
# plt.ylabel('loss')
# plt.xlabel('epochs')
# # plt.legend(loc='upper right')
# plt.legend()
# plt.show()



# acc 스코어 : 0.8221842298512942

