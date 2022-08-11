import numpy as np
aaa = np.array([-10,2,3,4,5,6,700,7,8,9,10,11,12,50,10])
aaa = aaa.reshape(-1,1) # (13,1)
from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.3)
# 이상치의 위치를 찾아준다.contamination은 범위를 지정


outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)

