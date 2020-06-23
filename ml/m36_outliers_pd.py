# 200623_26
# m36 판다스형


import numpy as np
import pandas as pd

# iloc
def outliers_iloc(data_out):
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

# loc
def outliers_loc(data_out):
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

# how to edit numpy & pandas outliers
def outliers(data_out):
    out = []
    count = 0
    if str(type(data_out))== str("<class 'numpy.ndarray'>"):
        for col in range(data_out.shape[1]):
            data = data_out[:,col]
            print(data)

            quartile_1, quartile_3 = np.percentile(data,[25,75])
            print("1사분위 : ",quartile_1)
            print("3사분위 : ",quartile_3)
            iqr = quartile_3 - quartile_1
            lower_bound = quartile_1 - (iqr*1.5)
            upper_bound = quartile_3 + (iqr*1.5)
            out_col = np.where((data>upper_bound)|(data<lower_bound))
            print(out_col)
            data = data[out_col]
            print(f"{col+1}번째 행렬의 이상치 값: ", data)
            out.append(out_col)
            count += len(out_col)

    if str(type(data_out))== str("<class 'pandas.core.frame.DataFrame'>"):
        i=0
        print(data_out.columns)
        for col in data_out.columns:
            data = data_out[col].values
            print(data)
            print(type(data))
            quartile_1, quartile_3 = np.percentile(data,[25,75])
            # print("1사분위 : ",quartile_1)
            # print("3사분위 : ",quartile_3)
            iqr = quartile_3 - quartile_1
            lower_bound = quartile_1 - (iqr*1.5)
            upper_bound = quartile_3 + (iqr*1.5)
            out_col = np.where((data>upper_bound)|(data<lower_bound))
            print('===out_col===')
            print(out_col)
            print('===out_col[0]===')
            print(out_col[0], i)
            data_out.iloc[out_col[0],i]=np.nan
            data = data[out_col]
            print(f"'{col}'의 이상치값: ", data)
            print(type(data))
            i+=1
    return data_out


### pd 데이터
df = pd.DataFrame({'a':[1,2,1000,4,5],
                   'b':[2,3,4,5,6000]})
print(df)
print(df.shape)

print('====a1====')

a1 = outliers(df)
print(a1)
print(type(a1))

print('====a2====')

# a2 = outliers2(df)
# print(a2)

print('====--====')
