import pandas as pd 
import numpy as np 

data = pd.DataFrame([[2,np.nan,6,8,10],
                     [2, 4,np.nan,8,np.nan],
                     [2,4,6,8,10],
                     [np.nan,4,np.nan,8,np.nan,],
                     ])

# print(data)
data = data.transpose()
data.columns = ['x1','x2','x3','x4']
print(data.shape)
print(data)

# 결측치 확인
print(data.isnull())
print(data.isnull().sum())  
# x1    2
# x2    2
# x3    2
# x4    3
print(data.info())

#1. 결측치 삭제
print("==================결측치 삭제=================")
print(data.dropna()) # 기본값은 행 기준 삭제
#     x1   x2   x3   x4
# 3  8.0  8.0  8.0  8.0
print(data.dropna(axis=1)) # 열 기준 삭제
#      x3
# 0   2.0
# 1   4.0
# 2   6.0
# 3   8.0
# 4  10.0
print("==================결측치 처리 : mean()=================")
#2-1  특정값- 평균 ->전체 평균이 아닌 각 컬럼 데이터로 대치
means = data.mean()
print("평균 :", means)
data2 = data.fillna(means)
# print(data2)
#      x1        x2    x3   x4
# 0   2.0  2.000000   2.0  6.0
# 1   6.5  4.000000   4.0  4.0
# 2   6.0  4.666667   6.0  6.0
# 3   8.0  8.000000   8.0  8.0
# 4  10.0  4.666667  10.0  6.0
print("==================결측치 처리 : median()=================")
#2-2  특정값- 평균 ->전체 평균이 아닌 각 컬럼 데이터로 대치
medians = data.median()
print("중위값 :", medians)
# 중위값 : 
# x1    7.0
# x2    4.0
# x3    6.0
# x4    6.0
data3 = data.fillna(medians)
print(data3)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  6.0
# 1   7.0  4.0   4.0  4.0
# 2   6.0  4.0   6.0  6.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  4.0  10.0  6.0

print("==================결측치 처리 : ffill()=================")
#2-3  특정값-ffill,bfill  (앞 행,뒷 행의 값으로 대치) 
# ffill은 시작 행이 nan 일경우 채울 수 없다.
# bfill은 마지막 행이 nan 일경우 채울 수 없다.
data4 = data.fillna(method='ffill')
print(data4)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   2.0  4.0   4.0  4.0
# 2   6.0  4.0   6.0  4.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  8.0  10.0  8.0
print("==================결측치 처리 : bfill()=================")
data5 = data.fillna(method='bfill')
print(data5)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  4.0
# 1   6.0  4.0   4.0  4.0
# 2   6.0  8.0   6.0  8.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

#2-4  특정값-임의의 값으로 채우기
print("==================결측치 처리 : 임의의 값으로 채우기=================")
data6 = data.fillna(value = 77777)
print(data6)
#         x1       x2    x3       x4
# 0      2.0      2.0   2.0  77777.0
# 1  77777.0      4.0   4.0      4.0
# 2      6.0  77777.0   6.0  77777.0
# 3      8.0      8.0   8.0      8.0
# 4     10.0  77777.0  10.0  77777.0

################### 특정 컬럼만!!!!!!! #######################
means = data['x1'].mean()
data['x1'] = data['x1'].fillna(means)
meds = data['x2'].median()
data['x2'] = data['x2'].fillna(meds)
data['x4'] = data['x4'].fillna(7777)
print(data)
#x1~x4까지 각 컬럼들을 채울 옵션을 설정하여 처리할 수 있다.
#      x1   x2    x3      x4
# 0   2.0  2.0   2.0  7777.0
# 1   6.5  4.0   4.0     4.0
# 2   6.0  4.0   6.0  7777.0
# 3   8.0  8.0   8.0     8.0
# 4  10.0  4.0  10.0  7777.0


