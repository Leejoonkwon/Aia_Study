from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import numpy as np

datasets = load_boston()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(
    x,  y, train_size= 0.7 , random_state=66
)
print(np.min(x)) # 0.0
print(np.max(x)) # 711.0
x = (x- np.min(x))/(np.max(x) - np.min(x)) # 최댓값에서 최솟값을 -한 값으로 .
print(x[:10])


print("=================")
print(x[:3])


