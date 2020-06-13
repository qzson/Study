# 20-06-12_23 금요일

# 데이콘 3 충돌체 감지
# 기본 시작 노트 (lb:0.64676)
# https://dacon.io/competitions/official/235614/codeshare/1133?page=1&dtype=recent&ptype=pub


import pandas as pd
import numpy as np

# 데이터 불러오기
train_features = pd.read_csv('./data/dacon/comp3/train_features.csv')
train_target = pd.read_csv('./data/dacon/comp3/train_target.csv')
test_features = pd.read_csv('./data/dacon/comp3/test_features.csv')

train_features.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1050000 entries, 0 to 1049999
# Data columns (total 6 columns):
#  #   Column  Non-Null Count    Dtype
# ---  ------  --------------    -----
#  0   id      1050000 non-null  int64
#  1   Time    1050000 non-null  float64
#  2   S1      1050000 non-null  float64
#  3   S2      1050000 non-null  float64
#  4   S3      1050000 non-null  float64
#  5   S4      1050000 non-null  float64
# dtypes: float64(5), int64(1)
# memory usage: 48.1 MB

print(f'\n<train_features.describe> \n{train_features.describe()}')
# <train_features.describe>
#                  id          Time            S1            S2            S3            S4
# count  1.050000e+06  1.050000e+06  1.050000e+06  1.050000e+06  1.050000e+06  1.050000e+06
# mean   1.399500e+03  7.480000e-04 -4.050983e+02 -4.050983e+02 -1.334343e+03 -1.605664e+03
# std    8.082907e+02  4.330114e-04  2.753174e+05  2.753174e+05  2.655351e+05  3.026970e+05
# min    0.000000e+00  0.000000e+00 -5.596468e+06 -5.596468e+06 -2.772952e+06 -6.069645e+06
# 25%    6.997500e+02  3.720000e-04 -7.426321e+04 -7.426321e+04 -7.855488e+04 -7.818371e+04
# 50%    1.399500e+03  7.480000e-04  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00
# 75%    2.099250e+03  1.124000e-03  7.391142e+04  7.391142e+04  7.295836e+04  7.665808e+04
# max    2.799000e+03  1.496000e-03  3.865086e+06  3.865086e+06  3.655237e+06  3.687344e+06 

# >> S1, S2는 동일한 분포를 갖는 데이터
# >> S1, S2, S3, S4는 중위값을 0으로 갖는 데이터

print(f'\n<test_features.describe> \n{test_features.describe()}')
# <test_features.describe>
#                   id           Time            S1            S2            S3            S4
# count  262500.000000  262500.000000  2.625000e+05  2.625000e+05  2.625000e+05  2.625000e+05
# mean     3149.500000       0.000748 -2.172298e+02 -1.842608e+02 -1.208247e+02 -8.578727e+02
# std       202.072773       0.000433  2.303438e+05  2.285628e+05  2.282941e+05  2.691352e+05
# min      2800.000000       0.000000 -3.027980e+06 -2.783507e+06 -2.399706e+06 -5.163090e+06
# 25%      2974.750000       0.000372 -7.873856e+04 -7.896356e+04 -8.323576e+04 -7.888264e+04
# 50%      3149.500000       0.000748  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00
# 75%      3324.250000       0.001124  7.698237e+04  7.719620e+04  8.165685e+04  7.845508e+04
# max      3499.000000       0.001496  3.022443e+06  2.877832e+06  2.486714e+06  4.305746e+06

# >> train 과는 다르게 S1, S2의 분포가 다르다

print('\ntrain_features.grouby(id) \n', train_features.groupby(['id']).count())
# train_features.grouby
#        Time   S1   S2   S3   S4
# id
# 0      375  375  375  375  375
# 1      375  375  375  375  375
# 2      375  375  375  375  375
# 3      375  375  375  375  375
# 4      375  375  375  375  375
# ...    ...  ...  ...  ...  ...
# 2795   375  375  375  375  375
# 2796   375  375  375  375  375
# 2797   375  375  375  375  375
# 2798   375  375  375  375  375
# 2799   375  375  375  375  375

# >> 모든 id는 375개의 값을 가진다(375*2,800 = 1,050,000)

print('\ntest_features.groupby(id) \n', test_features.groupby(['id']).count())
# test_features.groupby(id)
#        Time   S1   S2   S3   S4
# id
# 2800   375  375  375  375  375
# 2801   375  375  375  375  375
# 2802   375  375  375  375  375
# 2803   375  375  375  375  375
# 2804   375  375  375  375  375
# ...    ...  ...  ...  ...  ...
# 3495   375  375  375  375  375
# 3496   375  375  375  375  375
# 3497   375  375  375  375  375
# 3498   375  375  375  375  375
# 3499   375  375  375  375  375

# >> 모든 id는 375개의 값을 가진다(375*700 = 262,500)


print('\ntrain_features.groupby(id,time) \n', train_features.groupby(['id','Time']).count())
# train_features.groupby(id,time)
#                 S1  S2  S3  S4
# id   Time
# 0    0.000000   1   1   1   1
#      0.000004   1   1   1   1
#      0.000008   1   1   1   1
#      0.000012   1   1   1   1
#      0.000016   1   1   1   1
# ...            ..  ..  ..  ..
# 2799 0.001480   1   1   1   1
#      0.001484   1   1   1   1
#      0.001488   1   1   1   1
#      0.001492   1   1   1   1
#      0.001496   1   1   1   1

# >> 모든 id는 동일한 시간 간격을 가지고 0~0.001496까지 기록되어있다.(4*(375-1) = 1496)
