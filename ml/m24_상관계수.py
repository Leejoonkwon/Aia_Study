import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris()
print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x = datasets['data']
y = datasets['target']

df = pd.DataFrame(x, columns=[['sepal length', 'sepal width', 'petal length', 'petal width']])
# print(df)

df['Target'] = y # y라는 컬럼을 추가하자.
print(df) # [150 rows x 5 columns]

# 사이킷런에서 제공하는 데이터셋을 판다스로 바꿔서 컬럼명까지 지정
print("========================== 상관계수 히트 맵 =========================")
print(df.corr()) # 상관계수를 확인해보자. 각 컬럼별 상관관계를 확인할 수 있다.
#              sepal length sepal width petal length petal width    Target
# sepal length     1.000000   -0.117570     0.871754    0.817941  0.782561      
# sepal width     -0.117570    1.000000    -0.428440   -0.366126 -0.426658      
# petal length     0.871754   -0.428440     1.000000    0.962865  0.949035      
# petal width      0.817941   -0.366126     0.962865    1.000000  0.956547      
# Target           0.782561   -0.426658     0.949035    0.956547  1.000000      

# 데이터를 신뢰할 수 있다는 가정하에, 제거할 컬럼을 확인 가능

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)

plt.show()