# Polynomial Features = 다항 회귀
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(8).reshape(4, 2)
# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]
print(x.shape) #(4, 2)

pf = PolynomialFeatures(degree=2)

x_pf = pf.fit_transform(x)
##############degree=2 
# [[ 1.  0.  1.  0.  0.  1.]
#  [ 1.  2.  3.  4.  6.  9.]
#  [ 1.  4.  5. 16. 20. 25.]
#  [ 1.  6.  7. 36. 42. 49.]]
# 첫번째자리 0은 고정 새로운 자릿수는[x,y]였다면 
# [1,x,y,x의제곱,x*y,y의 제곱,]

print(x_pf)
print(x_pf.shape) #(4, 6)
##########################################
x = np.arange(12).reshape(4, 3)
print(x)
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]
print(x.shape) #(4, 3)
# [ 1 , x, y, z, x제곱,x*y,x*z,y제곱,y*z,z제곱]
# [[  1.   0.   1.   2.   0.   0.   0.   1.   2.   4.]
#  [  1.   3.   4.   5.   9.  12.  15.  16.  20.  25.]
#  [  1.   6.   7.   8.  36.  42.  48.  49.  56.  64.]
#  [  1.   9.  10.  11.  81.  90.  99. 100. 110. 121.]]
pf = PolynomialFeatures(degree=2)
x_pf = pf.fit_transform(x)
print(x_pf)
print(x_pf.shape) #(4, 10)
##########################################
x = np.arange(8).reshape(4, 2)
# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]
print(x.shape) #(4, 2)

pf = PolynomialFeatures(degree=3)

x_pf = pf.fit_transform(x)
# ##############degree=3
#  [1 , x, y, x제곱, x*y, y제곱,x제곱*y,x*y제곱,x제곱*y제곱] 
# [[  1.   0.   1.   0.   0.   1.   0.   0.   0.   1.]
#  [  1.   2.   3.   4.   6.   9.   8.  12.  18.  27.]
#  [  1.   4.   5.  16.  20.  25.  64.  80. 100. 125.]
#  [  1.   6.   7.  36.  42.  49. 216. 252. 294. 343.]]

print(x_pf)
print(x_pf.shape) #(4, 10)
