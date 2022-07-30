from tensorflow.keras.applications import VGG16
import numpy as np

model = VGG16()
model.summary()

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
# np.save('D:\study_data\_save\_npy\_train_x8.npy',arr=x_train)
# np.save('D:\study_data\_save\_npy\_train_y8.npy',arr=y_train)
# np.save('D:\study_data\_save\_npy\_test_x8.npy',arr=x_test)
# np.save('D:\study_data\_save\_npy\_test_y8.npy',arr=y_test)
x_train = np.load('D:\study_data\_save\_npy\_train_x8.npy')
y_train = np.load('D:\study_data\_save\_npy\_train_y8.npy')
x_test = np.load('D:\study_data\_save\_npy\_test_x8.npy')
y_test = np.load('D:\study_data\_save\_npy\_test_y8.npy')

model = VGG16(input_shape=(100, 100, 3), include_top=False, weights='imagenet')
output = model.output

# x = GlobalAveragePooling2D()(output)
# x = Dense(50, activation='relu')(x)
# output = Dense(21, activation='softmax', name='output')(x)

# model = Model(inputs=model.input, outputs=output)
# model.summary()

# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# hist = model.fit(x_train,y_train,epochs=50,verbose=2,
#                  validation_split=0.25,
#                  )

# #4. 평가,예측
# loss = model.evaluate(x_test, y_test)
# print('loss :', loss)
# y_predict = model.predict(x_test)
# y_predict = np.argmax(y_predict,axis=1)
# y_test = np.argmax(y_test,axis=1)

# print('y_predict :', y_predict.shape) #y_predict : (50,)
# from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_test, y_predict)
# print('acc 스코어 :', acc)

