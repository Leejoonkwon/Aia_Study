# 결측치 처리
#1. 행 또는 열 삭제
#2. 임의의 값 
    # 평균       : mean #평균의 함정을 조심해라!!!!
    # 중위       : median #아웃라이어에서 나온다!!! 평균이랑 다르다!
    # 0 대치     : fillna# 아주 bad한 idea 다!!!
    # 앞 값      : ffill
    # 뒷 값      : bfill
    # 특정값     :
    # 기타       :
#3. 보간 - interpolate    # 선형회귀방식과 같다.(linear)
#4. 모델 - predict => 결측을 제외한 값을 훈련시켜 결측인 행을 
# predict로 도출한 값 다시 데이터에 삽입한다.모델은 본인이 선택한다.
# x와 y 값에 구분을 두지 말고 결측치가 심한 컬럼 중 무엇을 Y로 두냐로 판단할 것 
#5. 부스팅계열 - 통상 결측,이상치에 대해 자유롭다.(믿거나 말거나 ㅋㅋ-선생님)
#->tree 계열도 비슷하지만 성능은 떨어진다.
import pandas as  pd  
import numpy as np
from datetime import datetime

dates = ['8/10/2022','8/11/2022','8/12/2022',
         '8/13/2022','8/14/2022']

dates = pd.to_datetime(dates)
print(dates)
print("==============================")

ts = pd.Series([2,np.nan,np.nan,8,10],index=dates)
print(ts)
# 2022-08-10     2.0
# 2022-08-11     NaN
# 2022-08-12     NaN
# 2022-08-13     8.0
# 2022-08-14    10.0
# dtype: float64
print("==============================")
ts = ts.interpolate()
print(ts)
# 2022-08-10     2.0
# 2022-08-11     4.0
# 2022-08-12     6.0
# 2022-08-13     8.0
# 2022-08-14    10.0
# dtype: float64


