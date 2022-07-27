# selenium의 webdriver를 사용하기 위한 import
from selenium import webdriver

# selenium으로 무엇인가 입력하기 위한 import
from selenium.webdriver.common.keys import Keys

# 페이지 로딩을 기다리는데에 사용할 time 모듈 import
import time

# 크롬드라이버 실행  (경로 예: '/Users/Roy/Downloads/chromedriver')
driver = webdriver.Chrome('C:\chromedriver.exe') 

import requests
from bs4 import BeautifulSoup
#크롬 드라이버에 url 주소 넣고 실행
driver.get('https://music.bugs.co.kr/genre/chart/etc/nost/total/day')

# 페이지가 완전히 로딩되도록 3초동안 기다림
time.sleep(3)

search_click = driver.find_element('xpath','//*[@id="CHARTday"]/table/tbody/tr[1]/td[4]/a')
# //*[@id="CHARTday"]/table/tbody/tr[2]/td[4]/a
# //*[@id="CHARTday"]/table/tbody/tr[3]/td[4]/a
# //*[@id="CHARTday"]/table/tbody/tr[100]/td[4]/a
search_click.click()

time.sleep(3)

search_click2 = driver.find_element('xpath','//*[@id="container"]/section[2]/div/div/xmp')
# //*[@id="container"]/section[2]/div/div/xmp
search_click2.click()
variable = driver.current_url

print(variable)
URL = variable
request = requests.get(URL)
html = request.text
soup = BeautifulSoup(html,'html.parser') #soup이라는 변수에는 BeautifulSoup() 함수를 이용해 html 즉 request의 text를. parser 분석하라는 뜻입니다.

print(soup.find('xmp').string) # 텍스트로 뽑아야함 
driver.back()
search_click = driver.find_element('xpath','//*[@id="CHARTday"]/table/tbody/'+'tr[2]/td[4]'+'/a')
# //*[@id="CHARTday"]/table/tbody/tr[2]/td[4]/a
# //*[@id="CHARTday"]/table/tbody/tr[3]/td[4]/a
# //*[@id="CHARTday"]/table/tbody/tr[100]/td[4]/a
search_click.click()

time.sleep(3)

search_click2 = driver.find_element('xpath','//*[@id="container"]/section[2]/div/div/xmp')
# //*[@id="container"]/section[2]/div/div/xmp
search_click2.click()
variable = driver.current_url
# //*[@id="container"]/section[2]/div/div/xmp
print(variable)
URL = variable
request = requests.get(URL)
html = request.text
soup = BeautifulSoup(html,'html.parser') #soup이라는 변수에는 BeautifulSoup() 함수를 이용해 html 즉 request의 text를. parser 분석하라는 뜻입니다.

print(soup.find('xmp').string) # 텍스트로 뽑아야함 
