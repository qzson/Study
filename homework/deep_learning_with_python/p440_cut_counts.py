# 14.4.4 구간 분할
# 이산 범위의 데이터를 분할하여 집계할 경우 사용

import pandas as pd
from pandas import DataFrame

attri_data1 = {'ID': ['100','101','102','103','104','106','108','110','111','113'],
               'city': ['서울','부산','대전','광주','서울','서울','부산','대전','광주','서울'],
               'birth_year': [1990,1989,1992,1997,1982,1991,1988,1990,1995,1981],
               'name': ['영이','순돌','짱구','태양','션','유리','현아','태식','민수','호식']}
attri_data_frame1 = DataFrame(attri_data1)
print(attri_data_frame1)
#     ID city  birth_year name
# 0  100   서울        1990   영이
# 1  101   부산        1989   순돌
# 2  102   대전        1992   짱구
# 3  103   광주        1997   태양
# 4  104   서울        1982    션
# 5  106   서울        1991   유리
# 6  108   부산        1988   현아
# 7  110   대전        1990   태식
# 8  111   광주        1995   민수
# 9  113   서울        1981   호식

# 분할 리스트 만들기
birth_year_bins = [1980, 1985, 1990, 1995, 2000]

# 구간 분할 실시
birth_year_cut_data = pd.cut(attri_data_frame1.birth_year, birth_year_bins)
print(birth_year_cut_data)

# 각 구간의 수 집계
ct = pd.value_counts(birth_year_cut_data)
print(ct)
# (1985, 1990]    4
# (1990, 1995]    3
# (1980, 1985]    2
# (1995, 2000]    1

# 각 구간 이름 정해주기
group_names = ['first1980', 'second1980', 'first1990', 'second1990']
birth_year_cut_data = pd.cut(attri_data_frame1.birth_year, birth_year_bins, labels = group_names)
ct1 = pd.value_counts(birth_year_cut_data)
print(ct1)
# second1980    4
# first1990     3
# first1980     2
# second1990    1
# Name: birth_year, dtype: int64

# 분할 수 전달
bycd = pd.cut(attri_data_frame1.birth_year, 2)
print(bycd)
#      (1989.0, 1997.0]
# 1    (1980.984, 1989.0]
# 2      (1989.0, 1997.0]
# 3      (1989.0, 1997.0]
# 4    (1980.984, 1989.0]
# 5      (1989.0, 1997.0]
# 6    (1980.984, 1989.0]
# 7      (1989.0, 1997.0]
# 8      (1989.0, 1997.0]
# 9    (1980.984, 1989.0]
# Name: birth_year, dtype: category
# Categories (2, interval[float64]): [(1980.984, 1989.0] < (1989.0, 1997.0]]]
