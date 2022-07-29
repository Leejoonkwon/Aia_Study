import os
from glob import glob
import pandas as pd

label_list = os.listdir('D:\study_data\\train/') # test내의 폴더들 for문에 넣기 위해 list형식으로 정의
# label_list = os.listdir('D:\test\choiminsik/') # test내의 폴더들 for문에 넣기 위해 list형식으로 정의

image_paths = [] # image_path들을 받아넣을 list정의

for label in label_list:      # 위에 정의된 label_list를 label로 하나씩 넣어 줌
    temp = glob(f'D:\study_data\\train/{label}/*') # for 문이 돌며 해당 경로 폴더 내에 있는 jpg파일들을 list 덩어리로 크게 가져옴
    
    for image_path in temp: # list를 다시 풀어서 각 각 하나의 인자(string형식) 으로 위에 정의된 image_paths에 입력
        image_paths.append(image_path) 

data = pd.DataFrame(image_paths)
data.to_csv('D:/study_data/test.csv')