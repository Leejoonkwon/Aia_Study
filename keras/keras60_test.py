import requests
from bs4 import BeautifulSoup

title = input("title:")
artist = input("artist:")

url = 'https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&query=' + title + artist + '가사'

response = requests.get(url)

if response.status_code == 200:
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    lyrics = soup.select_one('#main_pack > section.sc_new.sp_pmusic._au_music_collection._prs_mus_1st > div > div.group_music > ul > li:nth-child(1) > div.music_btn._lyrics_wrap > div > div.lyrics_txt._lyrics_txt')
    if lyrics == None:
        print("찾으시는 곡의 가사 정보를 찾을 수 없습니다.")
    else:
        lines = lyrics.select('p')
        for line in lines:
            print(line.get_text(), "\n")

else : 
    print(response.status_code)
