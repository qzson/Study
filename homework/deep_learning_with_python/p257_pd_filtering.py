# p257
''' 8.3.8 필터링 '''

import pandas as pd

data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}
df = pd.DataFrame(data)
print(df.index % 2 == 0)
print()
print(df[df.index % 2 == 0])

# [ True False  True False  True]

#       fruits  year  time
# 0      apple  2001     1
# 2     banana  2001     5
# 4  kiwifruit  2006     3


# p258
import numpy as np
import pandas as pd
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

# DataFrame을 생성하고 열을 추가합니다
df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

# 필터링을 사용하여 df의 "apple" 열이 5 이상이고, "kiwifruit" 열에서 5 이상의 값을 가진 행을 포함한 DataFrame을 df에 대입하세요
df = df.loc[df["apple"] >= 5]
df = df.loc[df["kiwifruit"] >= 5]
#df = df.loc[df["apple"] >= 5][df["kiwifruit"] >= 5] 라도 OK

print(df)

# apple  orange  banana  strawberry  kiwifruit
# 1      6       8       6           3         10
# 5      8       2       5           4          8
# 8      6       8       4           8          8
