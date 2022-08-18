# KMeans = 특정 라벨 부분의 최근접의 이웃에 평균
# 평균에 대한 속성이 생기는 것을 클러스터링이라고 한다.
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris,load_wine,load_breast_cancer 
from sklearn.cluster import KMeans # 대표적 비지도 학습 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

datasets = load_breast_cancer()

df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(df) #[178 rows x 13 columns]

kmeans = KMeans(n_clusters=2,random_state=10,
                # algorithm='auto'
                )
kmeans.fit(df)

print(kmeans.labels_)
print(datasets.target)
df['cluster'] = kmeans.labels_
df['target'] = datasets.target
# print('acc :',accuracy_score(datasets.target,kmeans.labels_))
print('acc :',accuracy_score(df['cluster'],df['target']))
# acc : 0.8541300527240774
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(
    df,kmeans.labels_,train_size=0.8,shuffle=True,
    random_state=123
)

model = RandomForestClassifier()
model.fit(x_train,y_train)
y_predict = model.predict(x_test)
print('model.score :',accuracy_score(y_test,y_predict))
