# p270~
''' 9.3 DataFrame 결합 '''
''' 9.3.1 결합 유형 '''
''' 9.3.2 내부 결합의 기본 '''

import numpy as np
import pandas as pd

data1 = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
         "year": [2001, 2002, 2001, 2008, 2006],
         "amount": [1, 4, 5, 6, 3]}
df1 = pd.DataFrame(data1)

data2 = {"fruits": ["apple", "orange", "banana", "strawberry", "mango"],
         "year": [2001, 2002, 2001, 2008, 2007],
         "price": [150, 120, 100, 250, 3000]}
df2 = pd.DataFrame(data2)

# df1, df2의 내용을 확인하세요
print(df1)
print()
print(df2)
print()
# df1 및 df2의 "fruits"를 Key로 내부 결합한 DataFrame을 df3에 대입하세요
df3 = pd.merge(df1, df2, on="fruits", how="inner")

# 출력합니다
# 내부 결합의 동작을 확인합시다
print(df3)

# amount      fruits  year
# 0       1       apple  2001
# 1       4      orange  2002
# 2       5      banana  2001
# 3       6  strawberry  2008
# 4       3   kiwifruit  2006

#        fruits  price  year
# 0       apple    150  2001
# 1      orange    120  2002
# 2      banana    100  2001
# 3  strawberry    250  2008
# 4       mango   3000  2007

#    amount      fruits  year_x  price  year_y
# 0       1       apple    2001    150    2001
# 1       4      orange    2002    120    2002
# 2       5      banana    2001    100    2001
# 3       6  strawberry    2008    250    2008


''' 9.3.3 외부 결합의 기본 '''

import numpy as np
import pandas as pd

data1 = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
         "year": [2001, 2002, 2001, 2008, 2006],
         "amount": [1, 4, 5, 6, 3]}

df1 = pd.DataFrame(data1)

data2 = {"fruits": ["apple", "orange", "banana", "strawberry", "mango"],
         "year": [2001, 2002, 2001, 2008, 2007],
         "price": [150, 120, 100, 250, 3000]}
df2 = pd.DataFrame(data2)

# df1, df2의 내용을 확인하세요
print(df1)
print()
print(df2)
print()

# df1 및 df2을 "fruits"를 Key로 외부 결합한 DataFrame을 df3에 대입하세요
df3 = pd.merge(df1, df2, on="fruits", how="outer")

# 출력합니다
# 외부 결합의 동작을 확인합시다
print(df3)

# amount      fruits  year
# 0       1       apple  2001
# 1       4      orange  2002
# 2       5      banana  2001
# 3       6  strawberry  2008
# 4       3   kiwifruit  2006

#        fruits  price  year
# 0       apple    150  2001
# 1      orange    120  2002
# 2      banana    100  2001
# 3  strawberry    250  2008
# 4       mango   3000  2007

#    amount      fruits  year_x   price  year_y
# 0     1.0       apple  2001.0   150.0  2001.0
# 1     4.0      orange  2002.0   120.0  2002.0
# 2     5.0      banana  2001.0   100.0  2001.0
# 3     6.0  strawberry  2008.0   250.0  2008.0
# 4     3.0   kiwifruit  2006.0     NaN     NaN
# 5     NaN       mango     NaN  3000.0  2007.0


''' 9.3.4 이름이 다른 열을 Key로 결합하기 '''

import pandas as pd

# 주문 정보입니다
order_df = pd.DataFrame([[1000, 2546, 103],
                         [1001, 4352, 101],
                         [1002, 342, 101]],
                        columns=["id", "item_id", "customer_id"])

# 고객 정보입니다
customer_df = pd.DataFrame([[101, "광수"],
                            [102, "민호"],
                            [103, "소희"]],
                           columns=["id", "name"])

# order_df를 바탕으로 "id"를 customer_df와 결합하여 order_df에 대입하세요
order_df = pd.merge(order_df, customer_df, left_on="customer_id", right_on="id", how="inner")

print(order_df)

# id_x  item_id  customer_id  id_y    name
# 0  1000     2546          103   103    소희
# 1  1001     4352          101   101    광수
# 2  1002      342          101   101    광수


''' 9.3.5 인덱스를 Key로 결합하기 '''

import pandas as pd

# 주문 정보입니다
order_df = pd.DataFrame([[1000, 2546, 103],
                         [1001, 4352, 101],
                         [1002, 342, 101]],
                        columns=["id", "item_id", "customer_id"])

# 고객 정보입니다
customer_df = pd.DataFrame([["광수"],
                            ["민호"],
                            ["소희"]],
                           columns=["name"])
customer_df.index = [101, 102, 103]

# customer_df를 바탕으로 "name"을 order_df와 결합하여 order_df에 대입하세요
order_df = pd.merge(order_df, customer_df, left_on="customer_id", right_index=True, how="inner")

print(order_df)

#      id  item_id  customer_id name
# 0  1000     2546          103   소희
# 1  1001     4352          101   광수
# 2  1002      342          101   광수
