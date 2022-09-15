import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F




USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10]) #이런식으로 정렬된 데이터보다 ex)3,4,1,2,9 와 같은 데이터로 하는 게
y = np.array([1,2,3,4,5,6,7,8,9,10]) #안전하다?
x_predict = np.array([11,12,13])    # (3,)

# [검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법 찾기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,train_size=0.7, 
                                                #    shuffle=True ,
                                                    random_state=138
)

#셔플의 기본값은 True
# torch tensor로 타입 변환
x_train = torch.FloatTensor(x_train).unsqueeze(-1).to(DEVICE)
x_test = torch.FloatTensor(x_test).unsqueeze(-1).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)
x_predict = torch.FloatTensor(x_predict).unsqueeze(-1).to(DEVICE)

# # StandardScaling 
# x_predict = (x_predict - torch.mean(x_train))/ torch.std(x_train)
# x_test = (x_test - torch.mean(x_train))/ torch.std(x_train)
# x_train = (x_train - torch.mean(x_train))/ torch.std(x_train)

# MinmaxScaling
x_predict = (x_predict - torch.min(x_train))/ (torch.max(x_train)-torch.min(x_train))
x_test = (x_test - torch.min(x_train))/ (torch.max(x_train)-torch.min(x_train))
x_train = (x_train - torch.min(x_train))/ (torch.max(x_train)-torch.min(x_train))

#2. 모델구성
model =nn.Sequential(
    nn.Linear(1,5),
    nn.Linear(5,7),
    nn.ReLU(),
    nn.Linear(7,5),
    nn.Linear(5,1)
).to(DEVICE)


#3, 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    
    loss = criterion(hypothesis,y)
    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 1000
for epoch in range(1 , epochs+1):
    loss = train(model, criterion, optimizer,x_train, y_train)
    print('epoch : {}, loss : {}'.format(epoch,loss))

#4. 평가,예측
def evaluate(model,criterion, x, y):
    model.eval()
    with torch.no_grad():
        y_predict = model(x)
        result = criterion(y_predict,y)
    return result.item()

loss2 = evaluate(model,criterion, x_test,y_test)
print('최종 loss : ',loss2)

results = model(x_predict).to(DEVICE)
print('result : ','\n',results.cpu().detach().numpy())
# 최종 loss :  1.6311030304194674e-09
# result :
#  [[11.00004 ]
#  [12.000052]
#  [13.000066]]




