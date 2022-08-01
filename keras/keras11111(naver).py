#!/usr/bin/env python3
#-*- codig: utf-8 -*-
import sys
import requests
import json
client_id = "v9xs2r7tpv"
client_secret = "6w5HVoHCy27oqwjNXyPguVynwgpATMLDW5eybnCv"
url="https://naveropenapi.apigw.ntruss.com/sentiment-analysis/v1/analyze"
headers = {
    "X-NCP-APIGW-API-KEY-ID": client_id,
    "X-NCP-APIGW-API-KEY": client_secret,
    "Content-Type": "application/json"
}
content = "늘 똑같은 일로 싸우다 지친 우리 끝날 때 됐나 봐\
너답지 않던 모습 더는\
지켜보기 힘들었어\
다시 주워 담기 힘든 말들\
쏟아내고 집에 돌아왔어\
그날에 나는 맘이 편했을까\
다신 안 보겠단 각오로
니가 못한 숙제 한 거잖아
나는 사랑이 필요해
이만큼 아프면 충분해
니가 핀 담배만큼 난 울었어
니가 가장 듣기 싫어했던
얘기들만 뱉어내고 왔어
그날에 나는 맘이 편했을까
다신 안 보겠단 각오로
니가 못한 숙제 한 거잖아
나는 사랑이 필요해
이만큼 아프면 충분해
니가 핀 담배만큼 난 울었어
상처받았다고 말하지 말아줘
나를 더욱더 사랑해줬더라면
아니 처음부터 만나지 말았다면
행복했을까
정말 널 미워해서 이랬을까
이렇게까지 해서라도
우릴 되돌리고 싶었는데
나를 떠나는 이유가
너는 필요했던 거니까
내가 그 이유를 만들어줄게
미안한 마음들 갖지 않도록"
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