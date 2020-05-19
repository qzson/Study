# 자료형

# 1. 리스트
a = [1,2,3,4,5]
# b = [1,2,3,a,b]
b = [1,2,3,'a','b']
c = [1,2,3,a,b]
print(b)
print(c,'\n')
# 리스트는 여러가지 자료형을 쓸 수 있다. BUT, numpy에는 딱 한가지 자료형만 사용해야한다.

print(a[0] + a[3],'\n')
# print(b[0] + b[3]) # 에러 : 형변환 필요
print(str(b[0])+b[3],'형변환')

print(type(a))
print(a[-2])
print(a[1:3],'\n')

a = [1,2,3,['a','b','c']]
print(a[1])
print(a[-1])
print(a[-1][1],'\n') # [큰 묶음][작은 묶음]


#1-2. 리스트 슬라이싱 ★
a = [1,2,3,4,5]
print(a[:2])


# 1-3. 리스트 더하기
a = [1,2,3]
b = [4,5,6]
print(a + b,'\n')

c = [7,8,9,10]
print(a+c)
print(a * 3)

# print(a[2] + 'hi') # 에러
print(str(a[2]) + 'hi')

f = '5'
# print(a[2]+f) # 에러
print(a[2]+int(f))
print('==========\n★ 중요★\n==========')


### 리스트 관련 함수 ###
a = [1,2,3]
a.append(4) # ★★★★ 절 대 잊 어 서 는 안 되 는 함 수 ★★★★
print(a)

# a = a.append(5) # 이렇게는 사용못한다. append는 그 자체로 사용해야함
# print(a)

a = [1,3,4,2]
a.sort()
print(a)

a.reverse()
print(a) # [4,3,2,1]

print(a.index(3)) # == a[3]
print(a.index(1)) # == a[1]

a.insert(0,7)
print(a)        # [7,4,3,2,1]
a.insert(3,3)
print(a)        # [7, 4, 3, 3, 2, 1]
a.remove(7) # 안에 있는 인자 값 지우기
print(a)        # [4, 3, 3, 2, 1]
a.remove(3) # 먼저 걸리는 놈 지우기
print(a)        # [4, 3, 2, 1]

# numpy 계산
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = a + b
print(c)               # [5 7 9]

print(a * 3)           # [3 6 9]

# numpy 식 계산은 다르다. (사람과 비슷)
# 장점 : 넘파이는 속도가 엄청 빠르고 부동소수점 계산?
# 단점 : 같은 타입만 사용 가능하다.