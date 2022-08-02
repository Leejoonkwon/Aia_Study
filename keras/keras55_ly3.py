
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
driver = webdriver.Chrome('C:\chromedriver.exe') 
import re
import pandas as pd

# title=driver.find_elements_by_class_name
driver.get("https://www.melon.com/chart/day/index.htm?classCd=GN0500")
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
print(number)    

LYRIC=[]

for i in number:
    driver.get("https://www.melon.com/song/detail.htm?songId=" + i) 
    lyric=driver.find_element(By.CLASS_NAME,"lyric")
    LYRIC.append(lyric.text)
df=pd.DataFrame({"제목":title2,"가수":singer2,"가사":LYRIC})
df.to_excel("멜론TOP50_가사5.xlsx",  encoding='utf-8')    
