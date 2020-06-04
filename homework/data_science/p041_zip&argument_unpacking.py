list1 = ['a','b','c']
list2 = [1,2,3]

# 실제 반복문이 시작되기 전까지는 묶어주지 않는다.
[pair for pair in zip(list1, list2)]    #[('a', 1), ('b', 2), ('c',3)]

pairs = [('a',1),('b',2),('c',3)]
letters, numbers = zip(*pairs)
print(letters, numbers)                 #('a', 'b', 'c') (1, 2, 3)

#이런 방식의 인자 헤체는 모든 함수에 적용할 수 있다.

def add(a, b): return a + b

add(1, 2)        #3
try:
    add([1, 2])
except TypeError:
    print("add expects two inputs")
add(*[1, 2])     #3