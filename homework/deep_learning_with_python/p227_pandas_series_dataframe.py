# 8.1.1 Series 와 DataFrame 의 데이터 확인
# 8.2.1 Series 생성
# 8.2.2 참조


import pandas as pd

fruits = {"orange": 2, "banana": 3}
print(pd.Series(fruits))

# orange    2
# banana    3
# dtype: int64

data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}
df = pd.DataFrame(data)
print(df)

#        fruits  year  time
# 0       apple  2001     1
# 1      orange  2002     4
# 2      banana  2001     5
# 3  strawberry  2008     6
# 4   kiwifruit  2006     3

print("=================")

# Series 용 라벨(인덱스)을 작성
index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

# Series 용 데이터를 대입
data = [10, 5, 8, 12, 3]

# Series 를 작성
series = pd.Series(data, index=index)

# 딕셔너리형을 사용하여 DataFrame용 데이터를 작성
data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}
df = pd.DataFrame(data)

print("Series 데이터")
print(series)
print("\n")
print("DataFrame 데이터")
print(df)

# Series 데이터
# apple         10
# orange         5
# banana         8
# strawberry    12
# kiwifruit      3
# dtype: int64

# DataFrame 데이터
#        fruits  year  time
# 0       apple  2001     1
# 1      orange  2002     4
# 2      banana  2001     5
# 3  strawberry  2008     6
# 4   kiwifruit  2006     3

import pandas as pd

fruits = {"banana": 3, "orange": 4, "grape": 1, "peach": 5}
series = pd.Series(fruits)
print(series[0:2])

# banana    3
# orange    4
# dtype: int64

import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)

# 인덱스 참조로 series의 2~4번째의 세 요소를 꺼내 items1에 대입하세요
items1 = series[1:4]

# 인덱스 값을 지정하는 방법으로 "apple", "banana", "kiwifruit"의 인덱스를 가지는 요소를 꺼내 items2에 대입하세요
items2 = series[["apple", "banana", "kiwifruit"]]
print(items1)
print()
print(items2)