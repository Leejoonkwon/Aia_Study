import numpy as np 
from sklearn.datasets import load_iris 
from sklearn.preprocessing import MinMaxScaler,StandardScaler 

#1. 데이터 
datasets = load_iris() 
x = datasets.data 
y = datasets.target   

from sklearn.preprocessing import MinMaxScaler,StandardScaler

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.8
       ,random_state=1234,shuffle=True)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train) 
# x_test = scaler.transform(x_test)

#2. 모델 
from sklearn.svm import LinearSVC, SVC 
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline 
from sklearn.decomposition import PCA # 주성분 분석 + 공부하기
# PCA를 통해서 feature를 조절한다.(압축의 개념)



# model = SVC()
# model = make_pipeline(MinMaxScaler(),SVC())
# model = make_pipeline(StandardScaler(),MinMaxScaler(),RandomForestClassifier())
model = make_pipeline(MinMaxScaler(),PCA(),RandomForestClassifier())
# 민맥스 스케일러를 통해서 스케일링 후 PCA를 통과하여 컬럼을 압축 그리고 RF를 통해 분류

# 모델 정의와 스케일링을 정의해주지 않아도  fit에서 fit_transform이 적용된다.
##################fit 까지는 fit_transform으로 스케일링 적용 
############# score,predict에서는 transform으로 적용된다.
#3. 훈련 
model.fit(x_train,y_train)

#4. 평가,예측
result = model.score(x_test,y_test)
# 모델 정의와 상관없이 model로 정의된 기능에 x_test에 transform이 적용된다

print('model.score :',result)
#3. 훈련 
# model.fit(x_train,y_train)

# #4. 평가,예측
# result = model.score(x_test,y_test)
# 모델 정의와 상관없이 model로 정의된 기능에 x_test에 transform이 적용된다

# print('model.score :',result)
