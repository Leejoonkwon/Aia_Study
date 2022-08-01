#!/usr/bin/env python3
#-*- codig: utf-8 -*-
import sys
import requests
import json
import pandas as pd
path = 'D:\study_data/' # ".은 현재 폴더"
df = pd.read_csv(path + 'music2.csv'  )
# print(df['lyric'][0]) #400

client_id = "v9xs2r7tpv"
client_secret = "6w5HVoHCy27oqwjNXyPguVynwgpATMLDW5eybnCv"
url="https://naveropenapi.apigw.ntruss.com/sentiment-analysis/v1/analyze"
headers = {
    "X-NCP-APIGW-API-KEY-ID": client_id,
    "X-NCP-APIGW-API-KEY": client_secret,
    "Content-Type": "application/json"
}
content = f"c의 값: {df['lyric'][54]}"

data = {
  "content": content
}
print(json.dumps(data, indent=4, sort_keys=True))
response = requests.post(url, data=json.dumps(data), headers=headers)
rescode = response.status_code
if(rescode == 200):
    print (response.text)
else:
    print("Error : " + response.text)
