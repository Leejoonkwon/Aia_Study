import numpy as np
import pandas as pd

# load dataset
from sklearn.datasets import load_iris
iris = load_iris()

target = iris['target']

num = np.unique(target, axis=0)
num = num.shape[0]

encoding = np.eye(num)[target]
print(encoding)
