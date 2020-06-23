# 200623_26
# m36 판다스형


import numpy as np
import pandas as pd

# iloc
def outliers(data_out):
    out = []
    for col in range(data_out.shape[1]):
        data = data_out.iloc[:, col]
        quartile_1, quartile_3 = data.quantile([.25, .75])
        print("1사 분위 :", quartile_1)
        print("3사 분위 :", quartile_3)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        ans = np.where((data > upper_bound) | (data < lower_bound))
        out.append(ans)
    return out

# ioc
def outliers2(data_out):
    out = []
    for col in data_out.columns:
        data = data_out.loc[:, col]
        quartile_1, quartile_3 = data.quantile([.25, .75])
        print("1사 분위 :", quartile_1)
        print("3사 분위 :", quartile_3)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        ans = np.where((data > upper_bound) | (data < lower_bound))
        out.append(ans)
    return out

### pd 데이터
df = pd.DataFrame({'a':[1,2,1000,4,5],
                   'b':[2,3,4,5,6000]})
print(df)
print(df.shape)

print('====a1====')

a1 = outliers(df)
print(a1)

print('====a2====')

a2 = outliers2(df)
print(a2)

print('====--====')
