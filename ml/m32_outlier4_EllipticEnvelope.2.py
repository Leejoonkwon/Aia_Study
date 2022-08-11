import numpy as np
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
                [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]])
# (2,13)
aaa = np.transpose(aaa) #(13,2)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.3)
# 이상치의 위치를 찾아준다.contamination은 범위를 지정
abc = aaa[:,0]
abc2 = aaa[:,1]
abc = abc.reshape(-1,1)
abc2 = abc2.reshape(-1,1)

outliers.fit(abc)
results1 = outliers.predict(abc)
print(results1)
# [-10,2 ,3 ,4 ,5 ,6 ,7 ,8, 9, 10, 11, 12, 50]
# [-1 -1  1  1  1  1  1  1  1  1  1 -1 -1]
outliers.fit(abc2)
results2 = outliers.predict(abc2)
print(results2)
# [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 420, 350]
# [ 1   1   -1      1   1   1       -1   1  -1   -1      1   1      1]   