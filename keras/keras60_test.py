
from dataclasses import replace
import time
import numpy as np      
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D,Flatten,Dense,MaxPool2D,Dropout

path = 'D:\study_data/' # ".은 현재 폴더"
df = pd.read_csv(path + 'music2.csv'
                       )

x_train = np.load('D:\study_data\_save\_npy\_train_x10.npy')
y_train = np.load('D:\study_data\_save\_npy\_train_y10.npy')
x_test = np.load('D:\study_data\_save\_npy\_test_x10.npy')
y_test = np.load('D:\study_data\_save\_npy\_test_y10.npy')

#2. 모델 

model = Sequential()
model.add(Conv2D(input_shape=(100, 100, 1), kernel_size=(3, 3), filters=32, padding='same', activation='relu'))
model.add(Conv2D(kernel_size=(3, 3), filters=64, padding='same', activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(kernel_size=(3, 3), filters=128, padding='same', activation='relu'))
model.add(Conv2D(kernel_size=(3, 3), filters=256, padding='valid', activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(21,activation='softmax'))
# model.summary()


model.load_weights("D:\study_data\_save\keras60_project4.h5")
start_time = time.time()
#3. 컴파일,훈련
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', 
                              verbose=1,restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
#                       save_best_only=True, 
#                       filepath="".join([filepath,'k24_', date, '_', filename])
#                     )
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# hist = model.fit(x_train,y_train,epochs=50,verbose=2,
#                  validation_split=0.25,
#                  callbacks=[earlyStopping]
#                  ,batch_size=500)
# model.save_weights("D:\study_data\_save\keras60_project4.h5")
# model.save_weights("./_save/keras23_5_save_weights1.h5")

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
end_time = time.time()-start_time
print("걸린 시간 :",end_time)
x_data = np.load('D:\study_data\_save\_npy\_train_x12.npy')
y_predict = model.predict(x_data)
y_predict = np.argmax(y_predict,axis=1)
print('y_predict :',y_predict) 
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
if y_predict[0]   ==   0  : print('분노한 표정-추천 노래 :',rap)
elif y_predict[0] ==   1  : print('혐오하는 표정-추천 노래 :',rap)
elif y_predict[0] ==   2  : print('공포스러워하는 표정-추천 노래 :',bal)
elif y_predict[0] ==   3  : print('행복해하는 표정-추천 노래 :',tro)
elif y_predict[0] ==   4  : print('무표정-추천 노래 :',soul)
elif y_predict[0] ==   5  : print('슬픈 표정-추천 노래 :',blues)
elif y_predict[0] ==   6  : print('놀라워하는 표정-추천 노래 :',dan)  
elif y_predict[0] ==   7  : print('불안한 표정-추천 노래 :',indy)
elif y_predict[0] ==   8  : print('감동받은 표정-추천 노래 :',tro)
elif y_predict[0] ==   9  : print('지루한 표정-추천 노래 :',indy)
elif y_predict[0] ==   10 : print('자신감넘치는  표정-추천 노래 :',rock)
elif y_predict[0] ==   11 : print('실망한 표정-추천 노래 :',blues)
elif y_predict[0] ==   12 : print('의심하는 표정-추천 노래 :',bal)  
elif y_predict[0] ==   13 : print('흥미로운 표정-추천 노래 :',rock)
elif y_predict[0] ==   14 : print('죄책감 표정-추천 노래 :',bal)
elif y_predict[0] ==   15 : print('질투 표정-추천 노래 :',blues)
elif y_predict[0] ==   16 : print('외로운 표정-추천 노래 :',indy)
elif y_predict[0] ==   17 : print('만족한 표정-추천 노래 :',dan)
elif y_predict[0] ==   18 : print('진지한 표정-추천 노래 :',soul)  
elif y_predict[0] ==   19 : print('억울한 표정-추천 노래 :',blues)
elif y_predict[0] ==   20 : print('승리한 표정-추천 노래 :',rock)  


#### 증폭 후 데이터 3만개 일때
# loss : [1.3046691417694092, 0.507749080657959]
# 걸린 시간 : 168.37035512924194
# y_predict : (6775,)
# acc 스코어 : 0.5077490774907749

##########VGG16 적용 데이터 2만장일 때
# loss : [0.3138679265975952, 0.9046236276626587]
# 걸린 시간 : 92.63064312934875
# y_predict : (4477,)
# acc 스코어 : 0.9046236318963592
##########VGG16 적용 데이터 3만장일 때
# loss : [0.2217244803905487, 0.9342288374900818]
# 걸린 시간 : 129.7907133102417
# y_predict : (6477,)
# acc 스코어 : 0.934228809634089
##########VGG16 적용 데이터 4만장일 때
# loss : [0.17270071804523468, 0.9478589296340942]
# 걸린 시간 : 169.3062162399292
# y_predict : (8477,)
# acc 스코어 : 0.9478589123510676






