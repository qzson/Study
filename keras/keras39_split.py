import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
a = np.array(range(1, 11)) # 1~ 10 : 
size = 5

def split_x(seq, size):
    aaa = []        # 는 리스트
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset]) # item for item in subset = 굳이 안넣고 subset 하면 간단.
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print("==============")
print(dataset)

''' <함수 풀이>
1. 함수 정의 : split_x(seq, size) 아래, dataset = split_x(a, size) 로 재사용된 것을 확인
2. a = seq임을 확인, 다시 위로 올라가서 for문 진입
3. len(x) : x의 길이(요소의 전체 개수)를 리턴하는 함수. 즉, len(a) = 10
4. for i in range(6) : range(6) = 0,1,2,3,4,5 = i
5. subset = seq[0:(0+5)] = a[0:5] = 0 ~ 4 (인덱스) = 1,2,3,4,5
6. aaa.append([subset]) = 1,2,3,4,5 = aaa 리스트에 추가
7. aaa = 1,2,3,4,5 리턴 후, i = 1,2,3,4,5 값 넣어 반복 진행
# (i = 1 ~ 5) 진행 결과
# 1:6 = 1~5 = 2,3,4,5,6
# 2:7 = 2~6 = 3,4,5,6,7
# 3:8 = 3~7 = 4,5,6,7,8
# 4:9 = 4~8 = 5,6,7,8,9
# 5:10 = 5~9 = 6,7,8,9,10
'''
# 주말 과제
# git - homework - folder - bookname - 공부하셈 (120p 까지)
# 실행이 되는 것들 만들라
# p052_histogram.py
# 정리를 해야겠다 싶은 것은 따로 정리를 해라