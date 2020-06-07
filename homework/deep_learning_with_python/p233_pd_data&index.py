''' 8.2.3 데이터와 인덱스 추출 '''

import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)

# series_values에 series의 데이터를 대입하세요
series_values = series.values

# series_index에 series의 인덱스를 대입하세요
series_index = series.index

print(series_values)
print(series_index)

# [10  5  8 12  3]
# Index(['apple', 'orange', 'banana', 'strawberry', 'kiwifruit'], dtype='object')


''' p.235 8.2.4 요소 추가 '''

import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]

series = pd.Series(data, index=index)

# 인덱스가 "pineapple" 이고, 데이터가 12인 요소를 series에 추가하세요 
pineapple = pd.Series([12], index=["pineapple"])
series = series.append(pineapple)
# series = series.append(pd.Series({"pineapple":12}))라도 OK 
print(series)

# apple         10
# orange         5
# banana         8
# strawberry    12
# kiwifruit      3
# pineapple     12
# dtype: int64


''' p235 8.2.5 요소 삭제 '''

import pandas as pd

index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]

# index와 data를 포함한 Series를 작성하여 series에 대입합니다
series = pd.Series(data, index=index)

# 인덱스가 strawberry인 요소를 제거해 series에 대입하세요
series = series.drop("strawberry")

print(series)

# apple        10
# orange        5
# banana        8
# kiwifruit     3
# dtype: int64