
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
driver = webdriver.Chrome('C:\chromedriver.exe') 
import re
import pandas as pd

# title=driver.find_elements_by_class_name
driver.get("https://www.melon.com/chart/index.htm")
title=driver.find_elements(By.CLASS_NAME,'ellipsis.rank01')

title2=[]
for i in title:
    title2.append(i.text)

del title2[0]
del title2[50:]

singer=driver.find_elements(By.CLASS_NAME,'ellipsis.rank02')
singer2=[]
for i in singer:
    singer2.append(i.text)


del singer2[0]
del singer2[50:]

songTagList = driver.find_elements(By.ID,'lst50')
number=[]
for i in songTagList:
    number.append(i.get_attribute('data-song-no'))
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')
lyric2=[]
for i in number:
    driver.get("https://www.melon.com/song/detail.htm?songId=" + i) 
    lyric=driver.find_elements(By.CLASS_NAME,"lyric on").text
    # lyrics = soup.select_one('#content > div > p.lyrics').get_text()
    lyric2.append(lyrics)
    
df=pd.DataFrame({"제목":title2,"가수":singer2,"가사":lyric2})
df.to_excel("멜론TOP50 가사.xlsx",  encoding='utf-8')    