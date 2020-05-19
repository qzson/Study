# print문과 format함수
a = '사과'
b = '배'
c = '옥수수'

print('선생님은 잘 생기셨다.\n')

print(a)
print(a, b)
print(a, b, c)

# ""와 ''의 사용상 차이는 없으나 통일성은 지켜줘야한다.

print("나는 {0}를 먹었다.".format(a))
print("나는 {0}와 {1}를 먹음.".format(a, b))
print("나는 {0}와 {1}와 {2}를 먹음".format(a, b, c))
print("난 {2}와 {0}와 {1} 먹".format(a,b,c),'\n')

print('나는', a, '를 먹었다.')
print('나는', a,'와', b,'를 먹음')
print('나는', a,'와', b,'와', c,'를 먹음''\n')

print('나는', a+ '를 먹었다.')
print('나는', a+ '와', b+ '를 먹음')
print('나는', a+ '와', b+ '와', c+ '를 먹음''\n')

print('나는 ', a, '를 먹었다.', sep='')
print('나는 ', a, '와 ', b, '를 먹음', sep='')
print('나는 ', a, '와 ', b, '와 ', c, '를 먹음', sep='')