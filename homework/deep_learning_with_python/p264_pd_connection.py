# p264~
''' 9.2 DataFrame 연결 '''
''' 9.2.1 인덱스나 컬럼이 일치하는 DataFrame 간의 연결 '''

import numpy as np
import pandas as pd

# 지정된 인덱스와 컬럼을 가진 DataFrame을 난수로 생성하는 함수입니다
def make_random_df(index, columns, seed):
    np.random.seed(seed)
    df = pd.DataFrame()
    for column in columns:
        df[column] = np.random.choice(range(1, 101), len(index))
    df.index = index
    return df

# 인덱스와 컬럼이 일치하는 DataFrame을 만듭니다
columns = ["apple", "orange", "banana"]
df_data1 = make_random_df(range(1, 5), columns, 0)
df_data2 = make_random_df(range(1, 5), columns, 1)

# df_data1과 df_data2를 세로로 연결하여 df1에 대입하세요
df1 = pd.concat([df_data1, df_data2], axis=0)

# df_data1과 df_data2를 가로로 연결하여 df2에 대입하세요
df2 = pd.concat([df_data1, df_data2], axis=1)

print(df1)
print(df2)
#    apple  orange  banana
# 1     45      68      37
# 2     48      10      88
# 3     65      84      71
# 4     68      22      89
# 1     38      76      17
# 2     13       6       2
# 3     73      80      77
# 4     10      65      72
#    apple  orange  banana  apple  orange  banana
# 1     45      68      37     38      76      17
# 2     48      10      88     13       6       2
# 3     65      84      71     73      80      77
# 4     68      22      89     10      65      72


''' 9.2.2 인덱스나 컬럼이 일치하지 않는 DataFrame 간의 연결 '''

import numpy as np
import pandas as pd

# 지정된 인덱스와 컬럼을 가진 DataFrame을 난수로 생성하는 함수입니다
def make_random_df(index, columns, seed):
    np.random.seed(seed)
    df = pd.DataFrame()
    for column in columns:
        df[column] = np.random.choice(range(1, 101), len(index))
    df.index = index
    return df


columns1 = ["apple", "orange", "banana"]
columns2 = ["orange", "kiwifruit", "banana"]

# 인덱스가 1,2,3,4이고 컬럼이 columns1인 DataFrame을 만듭니다
df_data1 = make_random_df(range(1, 5), columns1, 0)

# 인덱스가 1,3,5,7이고 컬럼이 columns2인 DataFrame을 만듭니다
df_data2 = make_random_df(np.arange(1, 8, 2), columns2, 1)

# df_data1과 df_data2를 세로로 연결하여 df1에 대입하세요
df1 = pd.concat([df_data1, df_data2], axis=0)

# df_data1과 df_data2를 가로로 연결하여 df2에 대입하세요
df2 = pd.concat([df_data1, df_data2], axis=1) 

print(df1)
print(df2)
#    apple  orange  banana  kiwifruit
# 1   45.0      68      37        NaN
# 2   48.0      10      88        NaN
# 3   65.0      84      71        NaN
# 4   68.0      22      89        NaN
# 1    NaN      38      17       76.0
# 3    NaN      13       2        6.0
# 5    NaN      73      77       80.0
# 7    NaN      10      72       65.0
#    apple  orange  banana  orange  kiwifruit  banana
# 1   45.0    68.0    37.0    38.0       76.0    17.0
# 2   48.0    10.0    88.0     NaN        NaN     NaN
# 3   65.0    84.0    71.0    13.0        6.0     2.0
# 4   68.0    22.0    89.0     NaN        NaN     NaN
# 5    NaN     NaN     NaN    73.0       80.0    77.0
# 7    NaN     NaN     NaN    10.0       65.0    72.0


''' 9.2.3 연결 시 라벨 지정하기 '''

import numpy as np
import pandas as pd

# 지정된 인덱스와 컬럼을 가진 DataFrame을 난수로 생성하는 함수입니다
def make_random_df(index, columns, seed):
    np.random.seed(seed)
    df = pd.DataFrame()
    for column in columns:
            df[column] = np.random.choice(range(1, 101), len(index))
    df.index = index
    return df

columns = ["apple", "orange", "banana"]
df_data1 = make_random_df(range(1, 5), columns, 0)
df_data2 = make_random_df(range(1, 5), columns, 1)

# df_data1과 df_data2를 가로로 연결하고, keys로 "X", "Y"를 지정하여 MultiIndex로 만든 뒤 df에 대입하세요
df = pd.concat([df_data1, df_data2], axis=1, keys=["X", "Y"])

# df의 "Y" 라벨 "banana"를 Y_banana에 대입하세요
Y_banana = df["Y", "banana"]

print(df)
print()
print(Y_banana)

# X                   Y              
#   apple orange banana apple orange banana
# 1    45     68     37    38     76     17
# 2    48     10     88    13      6      2
# 3    65     84     71    73     80     77
# 4    68     22     89    10     65     72

# 1    17
# 2     2
# 3    77
# 4    72
# Name: (Y, banana), dtype: int32

