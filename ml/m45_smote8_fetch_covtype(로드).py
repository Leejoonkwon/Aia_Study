# 실습
# 저장한 데이터 불러와서 실행 
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

x_train = np.load('C:\study\Study\_save\_npy/keras49_9_train_x.npy')
y_train = np.load('C:\study\Study\_save\_npy/keras49_9_train_y.npy')
x_test = np.load('C:\study\Study\_save\_npy/keras49_9_test_x.npy')
y_test = np.load('C:\study\Study\_save\_npy/keras49_9_test_y.npy')

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
import time
start_time = time.time()
#3. 훈련
model.fit(x_train,y_train)
end_time = time.time()-start_time
#4. 평가, 예측
y_predict = model.predict(x_test)

score = model.score(x_test,y_test)
from sklearn.metrics import accuracy_score,f1_score
# print('model.score :', score)
print('acc_score :',accuracy_score(y_test,y_predict))
# print('f1_score(macro) :',f1_score(y_test,y_predict,average='macro'))
# f1_score(macro) : 0.4397558777039733 이진 분류일 때 사용
print('f1_score(micro) :',f1_score(y_test,y_predict,average='micro'))
print('걸린 시간 :',end_time)  