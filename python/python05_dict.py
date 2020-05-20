# 200520 0900 ~
# 3. 딕셔너리 (중복 X)
# {키 : 벨류} = 쌍으로 되어 있다.
# {key : value}

a = {1: 'hi', 2: 'hello'}
print(a)                    # {1: 'hi', 2: 'hello'}
print(a[1])                 # hi
print(a.keys())             # dict_keys([1, 2])

b = {'hi':1,'hello':2}
print(b['hello'])           # 2
print(b.keys())             # dict_keys(['hi', 'hello'])
print('==========')


# 딕셔너리 요소 삭제
del a[1]                
print(a)                    # {2: 'hello'}
del a[2]
print(a)                    # {}
print('==========')


# 딕셔너리 중복
# a = {1:'a', 1:'b', 1:'c'}   # 키가 중복되는 상황 = 키는 중복되면 X
# print(a)                    # {1: 'c'} *가장 마지막 요소가 나온다. 덮어씌우는 형식.

b = {1:'a', 2:'a', 3:'a'}     # 벨류가 중복되는 상황 = 벨류는 값이니 중복도 상관없다.
print(b)                      # {1: 'a', 2: 'a', 3: 'a'}
print('==========')


# Key 숫자 아닐 경우
a = {'name' : 'yun', 'phone' : '010', 'birth' : '0511'}
print(a.keys())             # dict_keys(['name', 'phone', 'birth'])
print(a.values())           # dict_values(['yun', '010', '0511'])
print(type(a))              # <class 'dict'>
print(a.get('name'))        # yun
print(a['name'])            # yun
print(a.get('phone'))       # 010
print(a['phone'])           # 010