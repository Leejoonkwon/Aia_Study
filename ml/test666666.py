import pandas as pd
import numpy as np
import glob
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM,GRU,Dense,Dropout
from sklearn.model_selection import train_test_split

path = 'D:\study_data\_data\_csv\dacon_grow/'
all_input_list = sorted(glob.glob(path + 'train_input/*.csv'))
all_target_list = sorted(glob.glob(path + 'train_target/*.csv'))

test_input_list2 = sorted(glob.glob(path + 'test_input/*.csv'))
test_target_list2 = sorted(glob.glob(path + 'test_target/*.csv'))



train_input_list = all_input_list[:50]
train_target_list = all_target_list[:50]

val_input_list = all_input_list[50:]
val_target_list = all_target_list[50:]

# print(all_input_list)
print(val_input_list)
print(len(val_input_list))  # 8

def aaa(input_paths, target_paths): #, infer_mode):
    input_paths = input_paths
    target_paths = target_paths
    # self.infer_mode = infer_mode
   
    data_list = []
    label_list = []
    print('시작...')
    # for input_path, target_path in tqdm(zip(input_paths, target_paths)):
    for input_path, target_path in zip(input_paths, target_paths):
        input_df = pd.read_csv(input_path)
        target_df = pd.read_csv(target_path)
       
        input_df = input_df.drop(columns=['시간'])
        input_df = input_df.fillna(0)
       
        input_length = int(len(input_df)/1440)
        target_length = int(len(target_df))
        print(input_length, target_length)
       
        for idx in range(target_length):
            time_series = input_df[1440*idx:1440*(idx+1)].values
            # self.data_list.append(torch.Tensor(time_series))
            data_list.append(time_series)
        for label in target_df["rate"]:
            label_list.append(label)
    return np.array(data_list), np.array(label_list)
    print('끗.')

train_data, label_data = aaa(train_input_list, train_target_list) #, False)
val_data, val_target = aaa(val_input_list, val_target_list) #, False)
test_data,test_target = aaa(test_input_list2, test_target_list2) #, False)


# print(train_data[0])
print(len(train_data), len(label_data)) # 1607 1607
print(len(train_data[0]))   # 1440
print(label_data)   # 1440
print(train_data.shape, label_data.shape)   # (1607, 1440, 37) (1607,)
print(val_data.shape, val_target.shape)   # (206, 1440, 37) (206,)
# print(test_data.shape, submission_target.shape)   # (195, 1440, 37) (195,)


x_train,x_test,y_train,y_test = train_test_split(train_data,label_data,train_size=0.75,shuffle=False)

#2. 모델 구성
model = Sequential()
model.add(GRU(10,input_shape=(1440,37)))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.summary()


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam',metrics=['acc'])
# "".join은 " "사이에 있는 문자열을 합치겠다는 기능
hist = model.fit(x_train, y_train, epochs=5, batch_size=1000, 
                validation_data=(val_data, val_target),
                verbose=2,#callbacks = [earlyStopping]
                )

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)


# y_predict = model.predict(x_test)
# print(y_test.shape) #(152,)
# print(y_predict.shape) #(152, 13, 1)

# from sklearn.metrics import accuracy_score, r2_score,accuracy_score
# r2 = r2_score(y_test, y_predict)
# print('r2스코어 :', r2)


y_summit = model.predict(test_data)


 
test_target = pd.DataFrame(test_target,columns=['rate'])
test_target['rate'] = y_summit
# submission = submission.fillna(submission.mean())
# submission = submission.astype(int)
test_target.to_csv('D:\study_data\_data\_csv\dacon_grow\\test_target',index=False)
import os
import zipfile
os.chdir("D:\study_data\_data\_csv\dacon_grow\\test_target/")
submission = zipfile.ZipFile("D:\study_data\_data\_csv\dacon_grow\sample_submission.zip", 'w')
for path in test_target_list2:
    path = path.split('/')[-1]
    submission.write(path)
submission.close()





