
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator 
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.layers import Dense, Dropout, Input
from tensorflow.python.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import numpy as np
from keras import models, layers
from keras import Input
from keras.models import Model
x_train = np.load('D:\study_data\_save\_npy\_train_x5.npy')
y_train = np.load('D:\study_data\_save\_npy\_train_y5.npy')
x_test = np.load('D:\study_data\_save\_npy\_test_x5.npy')
y_test = np.load('D:\study_data\_save\_npy\_test_y5.npy')
# print(x_train.shape) #(28709, 48, 48, 6)
x_test2 = np.load('D:\study_data\_save\_npy\_train_x23.npy')
path = 'D:\study_data/' # ".은 현재 폴더"
df = pd.read_csv(path + 'music2.csv'
                       )


####
from keras.applications.resnet import ResNet50
pre_trained_Res = ResNet50(weights='imagenet',
                           include_top=False, input_shape=(70,70,3))
pre_trained_Res.trainable = True
pre_trained_Res.summary()
additional_model = models.Sequential()
additional_model.add(pre_trained_Res)
additional_model.add(Flatten())
additional_model.add(layers.Dense(128, activation='relu'))
additional_model.add(layers.Dense(128, activation='relu'))
additional_model.add(layers.Dense(64, activation='relu'))
additional_model.add(layers.Dense(7, activation='softmax'))


#3. 컴파일,훈련
import time
additional_model.load_weights("D:\study_data\_save\keras60_project10.h5")
start_time = time.time()
additional_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# hist = additional_model.fit(x_train,y_train,epochs=8,verbose=2,
#                  validation_split=0.25,batch_size=50)
                
# additional_model.save_weights("D:\study_data\_save\keras60_project10.h5")
                 
#4. 평가,예측
loss = additional_model.evaluate(x_test, y_test)
print('loss :', loss)
end_time = time.time()-start_time
print("걸린 시간 :",end_time)

y_predict = additional_model.predict(x_test2)
y_predict = np.argmax(y_predict,axis=1)
# y_test = np.argmax(y_test,axis=1)
print('y_predict :',y_predict[0])
# from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_test, y_predict)
# print('acc 스코어 :', acc)
from random import *
is1 = df['Genre'] == '발라드'
is2 = df['Genre'] == '댄스'
is3 = df['Genre'] == '랩,힙합'
is4 = df['Genre'] == '알앤비,소울'
is5 = df['Genre'] == '인디'
is6 = df['Genre'] == '록,메탈'
is7 = df['Genre'] == '트로트'
is8 = df['Genre'] == '포크,블루스'
is_1 = df[is1]
is_2 = df[is2]
is_3 = df[is3]
is_4 = df[is4]
is_5 = df[is5]
is_6 = df[is6]
is_7 = df[is7]
is_8 = df[is8]
# print(is_1,is_2)

i = randrange(50)  
bal = '{} - {}'.format(is_1['title'][i],is_1['artist'][i])
i = randrange(50,100)  
dan = '{} - {}'.format(is_2['title'][i],is_2['artist'][i])
i = randrange(100,150)  
rap = '{} - {}'.format(is_3['title'][i],is_3['artist'][i])
i = randrange(150,200)  
soul = '{} - {}'.format(is_4['title'][i],is_4['artist'][i])
i = randrange(200,250)  
indy = '{} - {}'.format(is_5['title'][i],is_5['artist'][i])
i = randrange(250,300)  
rock = '{} - {}'.format(is_6['title'][i],is_6['artist'][i])
i = randrange(300,350)  
tro = '{} - {}'.format(is_7['title'][i],is_7['artist'][i])
i = randrange(350,400)  
blues = '{} - {}'.format(is_8['title'][i],is_8['artist'][i])


# 결과를 출력합니다.
if y_predict[5]  ==   0  : print('분노 표정-추천 노래 :',rap,rock)
elif y_predict[5]==   1  : print('혐오 표정-추천 노래 :',rap,indy)
elif y_predict[5]==   2  : print('공포 표정-추천 노래 :',bal,blues)
elif y_predict[5]==   3  : print('행복 표정-추천 노래 :',tro,dan)
elif y_predict[5]==   4  : print('무표정-추천 노래 :',soul)
elif y_predict[5]==   5  : print('슬픔-추천 노래 :',blues)
elif y_predict[5]==   6  : print('놀람-추천 노래 :',dan)  

