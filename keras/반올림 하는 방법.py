# [실습]
import numpy as np
y =np.array( [0.9,00.1,0.2,0.4,0.58,0.61,0.53])

y =np.round(y,0)
print(y)
# [1. 0. 0. 0. 1. 1. 1.]
# round 함수는 (x,y) 기준으로 y에 입력한 정수의 소수점 자릿수로 반올림한다.
# round(x,1) 일 경우 나타나는 값 0.x이며 소수점 둘째자리에서 반올림해 첫째자리까지 나타낸다.
y = np.around(y)
print(y)
# [1. 0. 0. 0. 1. 1. 1.]
# around 함수는 0.5를 기준으로 반올림한다
y = np.where(y>0.5,0,1)
print(y)
# [0 1 1 1 0 0 0]
# where 함수는 위치 변환 등의 기능으로 자주 쓰이지만 조건부로 반올림 할 수 있다.
# (y>0.5,1,0) 라면 y 값이 0.5보다 클경우 1로 반환하고 아닐 경우 0으로 반환한다.
