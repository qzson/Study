# 함수의 가장 큰 목적 : 재사용

print('*** sum ***')
def sum1(a, b):          # def(정의), sum1(임의로), (c, d)(매개변수: 2개의 값을 받아들인다.)
    return a+b
a = 1
b = 2
c = sum1(a, b)
print(c)                 # 3


''' 곱셈, 나눗셈, 뺄셈 함수 만들기 '''

print('*** mul ***')
def mul1(a, b):
    return a*b
a=3
b=2
c = mul1(a, b)
print(c)


print('*** div ***')
def div1(a,b):
    return a/b
a=2
b=4
c=div1(a,b)
print(c)


print('*** sub ***')
def sub1(a,b):
    return a-b
a=3
b=4
c=sub1(a,b)
print(c)


print('*** param X ***')
# parameter(매개변수)가 없는 함수
def sayYeh():
    return 'Hi'
aaa=sayYeh()        # 변수를 받아들이지 안더라도 구성 가능하다.
print(aaa)


print('*** param 3개 ***')
def sum2(a, b, c):
    return a+b+c
a = 1
b = 2
c = 34
d = sum2(a, b, c)
print(d)