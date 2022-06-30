from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

# 데이터셋 로드
iris = load_iris()
df = pd.DataFrame(data= np.c_[iris.data, iris.target] , 
                  columns= ['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
from sklearn.model_selection import KFold

X = np.array(df.iloc[:, :-1]) # class 열 제외한 feature 열들 모음 -> array 변환
y = df['class']

# split 개수, 셔플 여부 및 seed 설정
kf = KFold(n_splits = 5, shuffle = True, random_state = 50)

# split 개수 스텝 만큼 train, test 데이터셋을 매번 분할
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
print(df)
asfasfasfasf