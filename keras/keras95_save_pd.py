# numpy 와 pd 데이터 떙겨오기거
# 경우에 따라, 헤더와 인덱스가 다르니 주의하자


import numpy as np
import pandas as pd

datasets = pd.read_csv('./data/csv/iris.csv',
                        index_col=None,
                        header=0, sep=',')
print(datasets)

## index_col 
# : None = index_column을 새로 생성          / 기존 data를 인덱스 column으로 주지 않겠다.
# : 숫자 = 해당 열을 index_column으로 보겠다. / data로 인식 X (data가 날아갈 여지 있음) 
# : default  = None

## header  
# : None = 0번째 행부터 data로 보겠다.  
# :  0   = 0번째 행은 header로 보겠다. / data로 인식X (data가 날아갈 여지 있음) 
# : defalut = 'infer'( 미루다 ) 

## sep
# : seperate : 각 열을 구분해주는 기준 
# : sep =',' : csv는 값이 ' , '로 나뉘어져 있다. 
# : default = ‘,’

#    index_column \  150    4  setosa  versicolor  virginica -> 0행 = header
#               0    5.1  3.5     1.4         0.2          0
#               1    4.9  3.0     1.4         0.2          0
#               2    4.7  3.2     1.3         0.2          0
#               3    4.6  3.1     1.5         0.2          0
#               4    5.0  3.6     1.4         0.2          0
#               ..   ...  ...     ...         ...        ...
#               145  6.7  3.0     5.2         2.3          2
#               146  6.3  2.5     5.0         1.9          2
#               147  6.5  3.0     5.2         2.0          2
#               148  6.2  3.4     5.4         2.3          2
#               149  5.9  3.0     5.1         1.8          2


#     index_column \   4  setosa  versicolor  virginica  -> 0행 = header
#               150
#               5.1  3.5     1.4         0.2          0
#               4.9  3.0     1.4         0.2          0
#               4.7  3.2     1.3         0.2          0
#               4.6  3.1     1.5         0.2          0
#               5.0  3.6     1.4         0.2          0
#               ..   ...     ...         ...        ...
#               6.7  3.0     5.2         2.3          2
#               6.3  2.5     5.0         1.9          2
#               6.5  3.0     5.2         2.0          2
#               6.2  3.4     5.4         2.3          2
#               5.9  3.0     5.1         1.8          2

print(datasets.shape) # index_col = None : (150, 5)   header = 0 
                      # index_col = 0    : (150, 4)   header = 0

                      # header = None    : (151, 5)   index_col = None

print(datasets.head()) # 위에서 5개만 보여주기
print(datasets.tail()) # 밑에서 5개

print('=============')

print(datasets.values) # pandas를 numpy로 바꾸는 것 (datasets.values)

aaa = datasets.values
print(type(aaa))       # numpy
 # 헤더와 인덱스를 제거하고 수치로만 된 데이터를 넘파이로 바꿀 수 있어야 한다.
 # loc, iloc 정도 별도로 공부해두도록

# np로 저장
np.save('./data/iris_pd.npy', arr=aaa)