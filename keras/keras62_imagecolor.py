import requests as req
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver import ActionChains
import time


url = 'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=403014132'

header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Trident/7.0;rv:11.0) like Gecko'}

melon = req.get(url, headers = header) # 멜론차트 웹사이트
melon_html = melon.text
melon_parse= bs(melon_html, 'html.parser')
driver = webdriver.Chrome('c:chromedriver.exe')
driver.get(url)
time.sleep(1)
def pageMove(path):
 driver.find_element_by_xpath(path).click()
 source = driver.page_source
 soup = bs(source,'html.parser' )
 lyric = soup.find_all('div',class_='lyric')
 song_name = soup.find_all('div',class_ = 'song_name')
 time.sleep(1)
 driver.back() # 이전 페이지 이동
 save_song(song_name,lyric) #파일 저장 함수
# 파일 저장 함수
def save_song(song_name, lyric):
     song_text = []
     for a in song_name:
       song_text.append(a.text.strip())
    real_song = song_text[0].replace("\t","").replace("\n","").replace("곡명","")
    full_lyric = []
    for i in lyric:
        full_lyric.append(i.text.strip())
    path = "C:/Users/LG/Desktop/won/대학원/music_new/"
    song_path = real_song+".txt"
    real_path = path+song_path
    a = str(real_path)
    f = open(a ,'w',-1,"utf-8")
    f.write(full_lyric[0])
    f.close()
    
action = ActionChains(driver)
for j in range(1,11):
 for i in range(1,51):
    time.sleep(5)
    pageMove('//*[@id="frm"]/div/table/tbody/tr['+str(i)+']/td[4]/div/a')
    driver.find_element_by_xpath('//*[@id="pageObjNavgation"]/div/span/a['+str(j)+']').click()
