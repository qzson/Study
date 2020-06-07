# p237
''' 8.2.6 필터링 '''

import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)

conditions = [True, True, False, False, False]
print(series[conditions])

# apple     10
# orange     5
# dtype: int64

# 필터링의 예2
print(series[series >= 5])

# apple         10
# orange         5
# banana         8
# strawberry    12
# dtype: int64


index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)

# series의 요소 중에서 5 이상 10 미만의 요소를 포함하는 Series를 만들어 series에 다시 대입하세요
series = series[series >= 5][series < 10]

print(series)

# orange    5
# banana    8
# dtype: int64


# p239
''' 8.2.7 정렬 '''

import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)

# series의 인덱스를 알파벳 순으로 정렬해 items1에 대입하세요
items1 = series.sort_index()

# series의 데이터 값을 오름차순으로 정렬해 items2에 대입하세요
items2 = series.sort_values()

print(items1)
print()
print(items2)

# apple         10
# banana         8
# kiwifruit      3
# orange         5
# strawberry    12
# dtype: int64

# kiwifruit      3
# orange         5
# banana         8
# apple         10
# strawberry    12
# dtype: int64
