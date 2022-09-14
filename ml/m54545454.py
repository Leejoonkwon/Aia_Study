import pandas as pd
import numpy as np
import glob
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LSTM,GRU
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False

train_data = np.load('D:/study_data/_save/_npy/train_data12.npy')
label_data = np.load('D:/study_data/_save/_npy/label_data12.npy')
val_data = np.load('D:/study_data/_save/_npy/val_data12.npy')
val_target = np.load('D:/study_data/_save/_npy/val_target12.npy')
test_data = np.load('D:/study_data/_save/_npy/test_data12.npy')
test_target = np.load('D:/study_data/_save/_npy/test_target12.npy')

# print(train_data[0])
# print(train_data.shape, label_data.shape) # (1607, 1440, 37) (1607,)
# print(val_data.shape, val_target.shape) # (206, 1440, 37) (206,)
# print(test_data.shape, test_target.shape)   # (195, 1440, 37) (195,)
col_check = pd.read_csv('D:\study_data\_data\_csv\dacon_grow\\train_input/'+'CASE_01.csv',encoding='utf-8')
# print(col_check.columns)
Index=['번호','내부온도관측치', '내부습도관측치', 'CO2관측치', 'EC관측치', '외부온도관측치', '외부습도관측치', '펌프상태',
       '펌프작동남은시간', '최근분무량', '일간누적분무량', '냉방상태', '냉방작동남은시간', '난방상태', '난방작동남은시간',
       '내부유동팬상태', '내부유동팬작동남은시간', '외부환기팬상태', '외부환기팬작동남은시간', '화이트 LED상태',
       '화이트 LED작동남은시간', '화이트 LED동작강도', '레드 LED상태', '레드 LED작동남은시간',
       '레드 LED동작강도', '블루 LED상태', '블루 LED작동남은시간', '블루 LED동작강도', '카메라상태', '냉방온도',
       '난방온도', '기준온도', '난방부하', '냉방부하', '총추정광량', '백색광추정광량', '적색광추정광량',
       '청색광추정광량']
m,n,r = train_data.shape
out_arr = np.column_stack((np.repeat(np.arange(m),n),train_data.reshape(m*(n),-1)))
out_df = pd.DataFrame(out_arr,columns=Index)
# out_df = out_df.drop(columns='0', axis=1)

# print(out_df['최근분무량'].value_counts())
# print(out_df['최근분무량'].unique(),len(out_df['최근분무량'].unique()))
import matplotlib.pyplot as plt
# out_df.plot(kind='scatter',x='내부온도관측치',y='내부습도관측치')
# out_df.boxplot(column=['내부온도관측치','내부습도관측치'])

# plt.show()
def outliers(data_out):
    quartile_1, q2 , quartile_3 = np.percentile(data_out,
                                               [25,50,75]) # percentile 백분위
    print("1사분위 : ",quartile_1) # 25% 위치인수를 기점으로 사이에 값을 구함
    print("q2 : ",q2) # 50% median과 동일 
    print("3사분위 : ",quartile_3) # 75% 위치인수를 기점으로 사이에 값을 구함
    iqr =quartile_3-quartile_1  # 75% -25%
    print("iqr :" ,iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound)|
                    (data_out<lower_bound))
Index1=['내부온도관측치', '내부습도관측치', 'CO2관측치', 'EC관측치', '외부온도관측치', '외부습도관측치', '펌프상태',
       '펌프작동남은시간', '최근분무량', '일간누적분무량', '냉방상태', '냉방작동남은시간', '난방상태', '난방작동남은시간',
       '내부유동팬상태', '내부유동팬작동남은시간', '외부환기팬상태', '외부환기팬작동남은시간', '화이트 LED상태',
       '화이트 LED작동남은시간', '화이트 LED동작강도', '레드_LED상태', '레드_LED작동남은시간',
       '레드 LED동작강도', '블루 LED상태', '블루 LED작동남은시간', '블루 LED동작강도', '카메라상태', '냉방온도',
       '난방온도', '기준온도', '난방부하', '냉방부하', '총추정광량', '백색광추정광량', '적색광추정광량',
       '청색광추정광량']                     
  
내부온도관측치_out_index= outliers(out_df['내부온도관측치'])[0]                        # 91405
내부습도관측치_out_index= outliers(out_df['내부습도관측치'])[0]                        # 0
CO2관측치_out_index= outliers(out_df['CO2관측치'])[0] # 0
EC관측치_out_index= outliers(out_df['EC관측치'])[0] #44
외부온도관측치_out_index= outliers(out_df['외부온도관측치'])[0] # 0
외부습도관측치_out_index= outliers(out_df['외부습도관측치'])[0] # 1
펌프상태_out_index= outliers(out_df['펌프상태'])[0] # 0
펌프작동남은시간_index= outliers(out_df['펌프작동남은시간'])[0] # 0
최근분무량_out_index= outliers(out_df['최근분무량'])[0]  # 0
일간누적분무량_out_index= outliers(out_df['일간누적분무량'])[0] # 0
냉방상태_out_index= outliers(out_df['냉방상태'])[0] # 38
냉방작동남은시간_out_index= outliers(out_df['냉방작동남은시간'])[0] # 0
난방상태_out_index= outliers(out_df['난방상태'])[0] # 0
난방작동남은시간_out_index= outliers(out_df['난방작동남은시간'])[0] # 0
내부유동팬상태_out_index= outliers(out_df['내부유동팬상태'])[0] # 0
내부유동팬작동남은시간_out_index= outliers(out_df['내부유동팬작동남은시간'])[0] # 89
외부환기팬상태_out_index= outliers(out_df['외부환기팬상태'])[0] # 138
외부환기팬작동남은시간_out_index= outliers(out_df['외부환기팬작동남은시간'])[0] # 138
화이트_LED상태_out_index= outliers(out_df['화이트 LED상태'])[0] # 138
화이트_LED작동남은시간_out_index= outliers(out_df['화이트 LED작동남은시간'])[0] # 138
화이트_LED동작강도_out_index= outliers(out_df['화이트 LED동작강도'])[0] # 138
레드_LED상태_out_index= outliers(out_df['레드 LED상태'])[0] # 138
레드_LED작동남은시간_out_index= outliers(out_df['레드 LED작동남은시간'])[0] # 138
레드_LED동작강도_out_index= outliers(out_df['레드 LED동작강도'])[0] # 138
블루_LED상태_out_index= outliers(out_df['블루 LED상태'])[0] # 138
블루_LED작동남은시간_out_index= outliers(out_df['블루 LED작동남은시간'])[0] # 138
블루_LED동작강도_out_index= outliers(out_df['블루 LED동작강도'])[0] # 138
카메라상태_out_index= outliers(out_df['카메라상태'])[0] # 138
냉방온도_out_index= outliers(out_df['냉방온도'])[0] # 138
난방온도_out_index= outliers(out_df['난방온도'])[0] # 138
기준온도_out_index= outliers(out_df['기준온도'])[0] # 138
난방부하_out_index= outliers(out_df['난방부하'])[0] # 138
냉방부하_out_index= outliers(out_df['냉방부하'])[0] # 138
총추정광량_out_index= outliers(out_df['총추정광량'])[0] # 138
백색광추정광량_out_index= outliers(out_df['백색광추정광량'])[0] # 138
적색광추정광량_out_index= outliers(out_df['적색광추정광량'])[0] # 138
청색광추정광량_out_index= outliers(out_df['청색광추정광량'])[0] # 138
print(len(내부온도관측치_out_index))

'''
lead_outlier_index = np.concatenate((#Age_out_index,                            # acc : 0.8650306748466258
                                    #  TypeofContact_out_index,                 # acc : 0.8920454545454546
                                    #  CityTier_out_index,                      # acc : 0.8920454545454546
                                     DurationOfPitch_out_index,               # acc : 0.9156976744186046
                                    #  Gender_out_index,                        # acc : 0.8920454545454546
                                    #  NumberOfPersonVisiting_out_index,        # acc : 0.8835227272727273
                                    #  NumberOfFollowups_out_index,             # acc : 0.8942598187311178
                                    #  ProductPitched_index,                    # acc : 0.8920454545454546
                                    #  PreferredPropertyStar_out_index,         # acc : 0.8920454545454546
                                    #  MaritalStatus_out_index,                 # acc : 0.8920454545454546
                                    #  NumberOfTrips_out_index,                 # acc : 0.8670520231213873
                                    #  Passport_out_index,                      # acc : 0.8920454545454546
                                    #  PitchSatisfactionScore_out_index,        # acc : 0.8920454545454546
                                    #  OwnCar_out_index,                        # acc : 0.8920454545454546
                                    #  NumberOfChildrenVisiting_out_index,      # acc : 0.8920454545454546
                                    #  Designation_out_index,                   # acc : 0.8869047619047619
                                    #  내부온도관측치_out_index                  # acc : 0.8932926829268293
                                     ),axis=None)
                              
print(len(lead_outlier_index)) #577

lead_not_outlier_index = []
for i in out_df.index:
    if i not in lead_outlier_index :
        lead_not_outlier_index.append(i)
out_df_clean = out_df.loc[lead_not_outlier_index]      
out_df_clean = out_df_clean.reset_index(drop=True)
# print(train_set_clean)
                         
x_train,x_test,y_train,y_test = train_test_split(train_data,label_data,train_size=0.91,shuffle=False)
     
#2. 모델 구성
model = Sequential()
model.add(GRU(100,return_sequences=True,input_shape=(1440,37)))
model.add(GRU(100))
model.add(Dense(256, activation='swish'))
model.add(Dense(128, activation='swish'))
model.add(Dense(64, activation='swish'))
model.add(Dense(32, activation='swish'))
model.add(Dense(1, activation='swish'))
model.summary()
import time
start_time = time.time()
#3. 컴파일, 훈련
from tensorflow.python.keras.callbacks import EarlyStopping,ReduceLROnPlateau
import time
es = EarlyStopping(monitor='val_loss',patience=10,mode='min',verbose=1)

from tensorflow.python.keras.optimizers import adam_v2
learning_rate = 0.01
optimizer = adam_v2.Adam(lr=learning_rate)

model.compile(loss='mae', optimizer='adam',metrics=['acc'])
# "".join은 " "사이에 있는 문자열을 합치겠다는 기능
hist = model.fit(x_train, y_train, epochs=200, batch_size=3000, 
                validation_data=(val_data, val_target),
                verbose=2,callbacks = [es]
                )
model.save_weights("C:\Study\_save/keras57_12_save_weights1.h5")


#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_predict,y_test)
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test,y_predict))
                      
                  

print(test_input_list2)
model.fit(train_data,label_data)
y_summit = model.predict(test_data)

path2 = 'D:\study_data\_data\_csv\dacon_grow\\test_target/' # ".은 현재 폴더"
targetlist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv','TEST_06.csv']
# [29, 35, 26, 32, 37, 36]
empty_list = []
for i in targetlist:
    test_target2 = pd.read_csv(path2+i)
    empty_list.append(test_target2)
    
empty_list[0]['rate'] = y_summit[:29]
empty_list[0].to_csv(path2+'TEST_01.csv')
empty_list[1]['rate'] = y_summit[29:29+35]
empty_list[1].to_csv(path2+'TEST_02.csv')
empty_list[2]['rate'] = y_summit[29+35:29+35+26]
empty_list[2].to_csv(path2+'TEST_03.csv')
empty_list[3]['rate'] = y_summit[29+35+26:29+35+26+32]
empty_list[3].to_csv(path2+'TEST_04.csv')
empty_list[4]['rate'] = y_summit[29+35+26+32:29+35+26+32+37]
empty_list[4].to_csv(path2+'TEST_05.csv')
empty_list[5]['rate'] = y_summit[29+35+26+32+37:]
empty_list[5].to_csv(path2+'TEST_06.csv')
# submission = submission.fillna(submission.mean())
# submission = submission.astype(int)

import os
import zipfile
filelist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv', 'TEST_06.csv']
os.chdir("D:\study_data\_data\_csv\dacon_grow/test_target")
with zipfile.ZipFile("D:\study_data\_data\_csv\dacon_grow/sample_submission.zip", 'w') as my_zip:
    for i in filelist:
        my_zip.write(i)
    my_zip.close()
print('Done')
print('R2 :', r2)
print('RMSE :', rmse)
end_time = time.time()-start_time
print('걸린 시간:', end_time)
'''


