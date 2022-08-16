# import requests
# from tqdm import tqdm
# from bs4 import BeautifulSoup
# import pandas as pd
# need_reviews_cnt = 1600

# url = 'https://movie.naver.com/movie/bi/mi/pointWriteFormList.naver?code=194196&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false&page='
# review_list = []

# # page를 1부터 1씩 증가하며 URL을 다음 페이지로 바꿈
# for page in tqdm(range(1, need_reviews_cnt+1)):
#     review_list.extend([[one_review.select_one('em').text, one_review.select('span')[3].text.strip(), one_review.select('em')[2].text]
#     for one_review in BeautifulSoup(requests.get(f"{url}{page}").text, 'html.parser').select('div.score_result > ul > li')])

# review_list
# print(review_list)



import requests
import csv
from tqdm import tqdm
from bs4 import BeautifulSoup

need_reviews_cnt = 1600

url = 'https://movie.naver.com/movie/bi/mi/pointWriteFormList.naver?code=213404&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false&page='
review_list = []

# page를 1부터 1씩 증가하며 URL을 다음 페이지로 바꿈
for page in tqdm(range(1, need_reviews_cnt+1)):

    review_list.extend([[one_review.select_one('em').text, one_review.select('span')[3].text.strip(), one_review.select('em')[2].text]
    for one_review in BeautifulSoup(requests.get(f"{url}{page}").text, 'html.parser').select('div.score_result > ul > li')])

review_list

with open ("../../study/samples.csv", "w", newline ="", encoding ='utf-8-sig') as f:
    write = csv.writer(f)
    write.writerows(review_list)

print(review_list)