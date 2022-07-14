import numpy as np

a = np.array(range(1,11)) #[1,2,3,4,5,6,7,8,9,10]
size = 5
def split_x(dataset, size): # def라는 예약어로 split_x라는 변수명을 아래에 종속된 기능들을 수행할 수 있도록 정의한다.
    aaa = []   #aaa 는 []라는 값이 없는 리스트임을 정의
    for i in range(len(dataset)- size + 1): # 6이다 range(횟수)
        #for문을 사용하여 반복한다.첫문장에서 정의한 dataset을 
        subset = dataset[i : (i + size)]
        #i는 처음 0에 개념 [0:0+size]
        # 0~(0+size-1인수 까지 )노출 
        aaa.append(subset) #append 마지막에 요소를 추가한다는 뜻
        #aaa는  []의 빈 리스트 이니 subset이 aaa의 []안에 들어가는 것
        #aaa 가 [1,2,3]이라면  aaa.append(subset)은 [1,2,3,subset]이 될 것이다.
    return np.array(aaa)    

print(len(a)- size + 1)

# # 문법
# # for (변수 in 객체) {
#     #객체의 모든 열거할 수 있는 프로퍼티의 개수만큼 반복적으로 실행하고자 하는 실행문;
# # #}
bbb = split_x(a, size)
# print(bbb) 
# # [[ 1  2  3  4  5]
# #  [ 2  3  4  5  6]
# #  [ 3  4  5  6  7]
# #  [ 4  5  6  7  8]
# #  [ 5  6  7  8  9]
# #  [ 6  7  8  9 10]]
# print(bbb.shape) #(6, 5)
x = bbb[:, :-1]
# x = bbb[-1]

y = bbb[:, -1]
# print(x,y)
print(bbb)
print(x)
print(x.shape)
print(np.array(range(100)))

