# 정수형
a = 1
b = 2
c = a + b
print(c)
d = a * b
print(d)
e = a / b
print(e)

print("\n----------\n")

# 실수형
a = 1.1
b = 2.2
c = a+b
print(c) #3.300000000003 // 부동소수점을 사용하기 때문

d = a*b
print(d) #2.420000000004

print("\n-두 실수가 같은지 판단하기-\n")

import math, sys
x = 0.1 + 0.2
math.fabs(x - 0.3) <= sys.float_info.epsilon
print(f"{x==0.3}")

print("\n----------\n")

e = a/b
print(e)

# 문자형
a = 'hel'
b = 'lo'
c = a + b
print(c)

# 문자 + 숫자
a = 123
b = '45'
# c = a + b >>> 타입 에러
# print(c)
# 이 상태로는 오류난다.
# 데이터를 받았을 때, 숫자가 아니고 한글일 때의 경우를 생각.

# 숫자를 문자변환 + 문자
a = 123
a = str(a)
print(a)
b = '45'
c = a + b
print(c,'\n')

a = 123
b = '45'
b = int(b)
c = a + b
print(c,"\n")

# 문자열 연산하기
a = 'abcdefgh'
print(a[0])
print(a[3])
print(a[5])
print(a[-1])
print(a[-2])
print(type(a),"\n")

b = 'xyz'
print(a + b,"\n")

# 문자열 인덱싱
a = 'Hello, Deep learning'
print(a[7]) # D
print(a[-1]) # g
print(a[-2]) # n
print(a[3:9]) # lo, De
print(a[3:-5]) # lo, Deep lea
print(a[:-1]) # Hello, Deep learnin
print(a[1:]) # ello, Deep learning
print(a[5:-4],"\n") # , Deep lear