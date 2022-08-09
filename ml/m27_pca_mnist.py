import numpy as np 
from sklearn.decomposition import PCA 
from keras.datasets import mnist 

(x_train, _), (x_test, _)=mnist.load_data()

# print(x_train.shape,x_test.shape) #(60000, 28, 28) (10000, 28, 28)

x = np.append(x_train,x_test,axis = 0)
# print(x.shape) #(70000, 28, 28)
#####################################
# [실습]
# pca를 통해 0.95 이상인 n_component는 몇개?
# 0.95 # 154
# 0.99 # 331
# 0.999 # 486
# 1.0 # 713
# 힌트  np.armax

x = x.reshape(70000,784)
pca = PCA(n_components=784) # 차원 축소 (차원=컬럼,열,피처)
x = pca.fit_transform(x) 
print(x.shape) # (506, 2)
pca_EVR = pca.explained_variance_ratio_ # PCA로 압축 후에 새로 생성된 피쳐 임포턴스를 보여준다.
print(sum(pca_EVR)) #0.999998352533973
print(pca_EVR)

cumsum = np.cumsum(pca_EVR)
# cumsum = np.argmax(cumsum,axis=1)
print(np.argmax(cumsum >=0.95)+1)   # 154
# print(np.argmax(cumsum >=0.99)+1)   # 331
# print(np.argmax(cumsum >=0.999)+1)  # 486
# print(np.argmax(cumsum >=1.0)+1)    # 713
# import matplotlib.pyplot as plt   
# plt.plot(cumsum)
# plt.grid()
# plt.show() # 그림을 그려서 컬럼이 손실되면 안되는 범위를 예상할 수 있다.
# # print(np.argwhere(cumsum >= 0.95)[0])