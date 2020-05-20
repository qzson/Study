# 200520 1000~

print('============')
### for문 ###
a = {'name' : 'yun', 'phone' : '010', 'birth' : '0511'}

for i in a.keys():
    print(i)

a = [1,2,3,4,5,6,7,8,9,10]
for i in a:                 # 인자의 개수 만큼 돌린다.
    i = i*i
    print(i)
# print('melong')
    # 들여쓰기에 따른 for문에 포함인지 아닌지 주의

for i in a:
    print(i)

print('============')
### while문 ###
'''
while 조건문 :              # 참일 동안 계속 돈다.
    수행할 문장
'''

print('============')
### if문 ###

if 1 :
    print('True')
else :
    print('False')

if 3 :
    print('True')
else :
    print('False')

if 0 :                  # False
    print('True')
else :
    print('False')

if -1 :
    print('True')
else :
    print('False')

print('============')
### 비교연산자 ###
# <, >, ==, !=, >=, <=

a = 1
if a == 1:
    print('출력')

money = 10000
if money >= 30000:
    print('한우먹자')
else:
    print('라면먹자')

print('====== 조건 연산자 ======')
### 조건연산자
# and, or, not
money = 20000
card = 1
if money >= 30000 or card == 1:
    print('한우먹자')
else:
    print('라면먹자')

print('====== for + if ======')
### for문과 섞어서 ###
jumsu = [90, 25, 67, 45, 80]
number = 0
for i in jumsu:
    if i >= 60:
        print("합격")
        number = number + 1         # 이거 이해 안감;;
    else:
        print("불합격")
    
print("합격인원 :", number, "명")

print('====== break ======')
# break, continue
jumsu = [90, 25, 67, 45, 80]
number = 0
for i in jumsu:
    if i < 30:
        break                       # break 당하면 가까운 for문을 날려버린다.
    if i >= 60:
        print("합격")
        number = number + 1

print("합격인원 :", number, "명")

print('====== continue ======')
jumsu = [90, 25, 67, 45, 80]
number = 0
for i in jumsu:
    if i < 60:
        continue                       # 컨티뉴 걸리면 아래 것 실행 안시키고 다시 반복
    if i >= 60:
        print("합격")
        number = number + 1

print("합격인원 :", number, "명")