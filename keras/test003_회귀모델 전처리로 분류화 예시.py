from sklearn.datasets import load_boston 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 
import pandas as pd 
import numpy as np 
boston = load_boston() 
boston_df = pd.DataFrame(boston.data, columns = boston.feature_names) 
boston_df['Price'] = boston.target 
y_target = boston_df['Price']  
X_data = boston_df.drop('Price', axis=1, inplace=False) 
X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size=0.2, random_state = 100) 
y_train = y_train.apply(lambda x : round(x)) # 추가 부분
y_test = y_test.apply(lambda x : round(x))   # 추가 부분

y_train[(y_train>0) & (y_train<=10)] = 0     # 추가 부분
y_train[(y_train>10) & (y_train<=20)] = 1
y_train[(y_train>20) & (y_train<=30)] = 2
y_train[(y_train>30) & (y_train<=40)] = 3
y_train[(y_train>40) & (y_train<=50)] = 4
#일반적으로 회귀 분석 모델은 훈련으로 추출된 데이터를 테스트 데이터에 대입 후 평가하여 차이에 대한 정도를 표시하지만
#위와 같이 범주형 데이터로 구분하여 분류 모델에 데이터로 활용해  정확도를 파악할 수 있다(accuracy)

y_test[(y_test>0) & (y_test<=10)] = 0         # 추가 부분
y_test[(y_test>10) & (y_test<=20)] = 1
y_test[(y_test>20) & (y_test<=30)] = 2
y_test[(y_test>30) & (y_test<=40)] = 3
y_test[(y_test>40) & (y_test<=50)] = 4


dt_clf = DecisionTreeClassifier() 
dt_clf.fit(X_train, y_train) 
pred = dt_clf.predict(X_test) 
print('accuracy: ', np.round(accuracy_score(y_test, pred),2))  