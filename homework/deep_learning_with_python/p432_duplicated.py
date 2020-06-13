# 14.4 데이터 요약
# 14.4.1 키별 통계량 산출

import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df.columns=['', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
            'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
            'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
print(df)
print(df['Alcohol'].mean())
# [178 rows x 14 columns]
# 13.000617977528083

# 14.4.2 중복 데이터
# 중복데이터 삭제

from pandas import DataFrame
dupli_data = DataFrame({'col1':[1,1,2,3,4,4,6,6],
                        'col2':['a','b','b','b','c','c','b','b']})
print(dupli_data)
#    col1 col2
# 0     1    a
# 1     1    b
# 2     2    b
# 3     3    b
# 4     4    c
# 5     4    c
# 6     6    b
# 7     6    b

dp = dupli_data.duplicated()
print(dp)
# 0    False
# 1    False
# 2    False
# 3    False
# 4    False
# 5     True
# 6    False
# 7     True
# dtype: bool

dd_dp = dupli_data.drop_duplicates()
print(dd_dp)
# col1 col2
# 0     1    a
# 1     1    b
# 2     2    b
# 3     3    b
# 4     4    c
# 6     6    b
