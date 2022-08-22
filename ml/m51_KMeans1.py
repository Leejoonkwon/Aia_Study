# KMeans = 특정 라벨 부분의 최근접의 이웃에 평균
# 평균에 대한 속성이 생기는 것을 클러스터링이라고 한다.
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris 
from sklearn.cluster import KMeans # 대표적 비지도 학습 
from sklearn.metrics import accuracy_score

datasets = load_iris()

df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(df) # [150 rows x 4 columns]

kmeans = KMeans(n_clusters=3,random_state=1234)
kmeans.fit(df)

print(kmeans.labels_)
print(datasets.target)
df['cluster'] = kmeans.labels_
df['target'] = datasets.target
# print('acc :',accuracy_score(datasets.target,kmeans.labels_))
print('acc :',accuracy_score(df['cluster'],df['target']))

# acc : 0.8933333333333333
