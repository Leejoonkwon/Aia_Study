# for문은 반복해야할 값을 일시에 처리할 때 사용한다.

#  문법 구조
# for [변수]  in [문자열,리스트,튜플] :
#     [수행 부분]
# 문자열,리스트,튜플 안에 있는 값들을 순서대로 [변수]에 넣어 [수행 부분]을 반복한다.

# for문은 중첩 사용도 가능하다.
# for [변수1] in [문자열1, 리스트1, 튜플1]:
#     [수행부분]
#     for [변수2] in [문자열2, 리스트2, 튜플3]:
#         [수행부분]

#예시 1
# arr = [1,2,3,4,5]
# for i in arr :
#     print(i) #1 2 3 4  5
#예시 2
# index = 0
# s = "BlockDMask"
# for a in s :
#     if a == 'l':
#         break 
#     index = index + 1
# print(index)    #1
#예시 3
# student = [180,170,164,199,182,172,177]
# for a in student :
#     if a > 170: # a값이 170 초과라면 다음으로 넘어간다.
#         continue
#     print(a) # 170 164
#예시 4
# for x in range(0,10) : 
#     # range(x,y)에서 x,y가 정수라면 x부터 y-1의 범위까지 숫자를 반복해 x에 넣는다
#     print(x) # 0  1 2 3 4 5 6 7 8 9
#예시 5
# result = 0
# for a in range(1,101): 범위 1~100
#     result = result + a
# print(result)    #5050    
    
    