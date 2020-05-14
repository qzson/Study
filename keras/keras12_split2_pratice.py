# split 개인 연습 파일

# 1. train:val:test = 6:2:2 데이터 분리
import numpy as np
x = np.array(range(1,101))
y = np.array(range(101,201))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = False,
    train_size = 0.6)

x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, shuffle = False,
    test_size = 0.5) # 40% 중 절반 = 20%

print(x_train)
print(x_val)
print(x_test)

# 2. 8:1:1
import numpy as np
x = np.array(range(1,101))
y = np.array(range(101,201))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = False,
    train_size = 0.8)

x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, shuffle = False,
    test_size = 0.5)

print(x_train)
print(x_val)
print(x_test)

# 3. 7:1:2
import numpy as np
x = np.array(range(1,101))
y = np.array(range(101,201))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = False,
    train_size = 0.7)

x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, shuffle = False,
    test_size = 1/3)
# 질문 : 그냥 1:2로 나누는 과정에서 나머지는 자동으로 분류되나요?
# 답변 : test_size 에서 test가 1/3으로 할당을 했으니, 나머지는 2/3으로 자동으로 연산

print(x_train)
print(x_val)
print(x_test)

# 4. 둘 중 하나만 써도 된다.
import numpy as np
x = np.array(range(1,101))
y = np.array(range(101,201))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = False,
    train_size = 0.6, test_size = 0.4)

# <구자님 부연설명 참조>
# train_size와 test_size를 둘 다 사용해도 되고, 둘 중 하나만 사용해도 됨
# 단, train_size + test_size = sum > 1 이면 에러 뜸
#                              sum < 1 이면 빠진 값 만큼 날아감
# ex) train_size = 0.6, test_size = 0.3 이면 sum = 0.9로 0.1만큼의 값이 사라진다.

# train_size = 0.6, test_size = 0.4 [가능]
# train_size = 0.6, test_size = 0.3 [나머지 10%는 어디루?]
# train_size = 0.6, test_size = 0.5 [Error 발생]

print(x_train)
print(x_test)