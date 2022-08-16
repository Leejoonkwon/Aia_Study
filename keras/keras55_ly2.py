import requests 
from urllib import parse 
from bs4 import BeautifulSoup 
import random
import time
random.uniform(0.2,1.2) 
base_url ='https://movie.naver.com/movie/point/af/list.naver?&page={}'
# url = base_url.format(1)
# res = requests.get(url)
comment_list = []

for page in range(1,101):
    url = base_url.format(page)
    res = requests.get(url)
    if res.status_code ==  200:
        soup = BeautifulSoup(res.text,'lxml')
        tds = soup.select('table.list_netizen > tbody > tr > td.title')
        for td in tds:
            movie_title = td.select_one('a.movie').text.strip()
            link = td.select_one('a.movie').get('href')
            link = parse.urljoin(base_url,link)
            score = td.select_one('div.list_netizen_score > em').text.strip()
            comment = td.select_one('br').next_sibling.strip()
            comment_list.append((movie_title,link))
        interval = round(random.uniform(0.2, 1.2),2)   
        time.sleep(interval) 
print('종료')        

import pandas as pd 
df = pd.DataFrame(comment_list,columns = ['영화제목','링크','평점','댓글'])
df.to_csv('naver_commen.csv',encoding='utf-8',index=False)