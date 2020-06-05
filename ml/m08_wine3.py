# 20-06-05

# 게마와 함께하는 wine예제

import pandas as pd
import matplotlib.pyplot as plt

# 와인 데이터 읽기
wine = pd.read_csv('./data/csv/winequality-white.csv', sep=';', header=0)

# groupby
count_data = wine.groupby('quality')['quality'].count() # (해당 컬럼 내) 퀄리티의 각 개체별로 숫자를 세겠다. (5가 몇개, 6이 몇개 ...)

print(count_data)
'''
quality
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
'''
# 데이터 부분이 5와 6에 치중되어 있다.

count_data.plot()
plt.show()

# 분류의 폭을 축소를 시켜보자 (그렇다면 분포의 폭이 기존보다 조금 더 나아지겠지?)