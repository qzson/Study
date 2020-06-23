# 200623_26
# 이상치 체크 함수
# 실습   : 행렬을 입력해서 컬럼별로 이상치 발견하는 함수를 구현하기
# 파일명 : m36_outliers.py


import numpy as np

def outliers(data_out):
    out = []
    for col in range(data_out.shape[1]):
        data = data_out[:, col]
        quartile_1, quartile_3 = np.percentile(data, [25, 75])
        print("1사 분위 :", quartile_1)
        print("3사 분위 :", quartile_3)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        ans = np.where((data > upper_bound) | (data < lower_bound))
        out.append(ans)
    return out

a = np.array([[1,2],[3,4],[10000,6],[7,5000],[90,100],[1,2],[3,4],[10000,6],[7,5000],[90,100]])
c = np.array([[1,2,3],[3,4,10000],[6,7,5000],[90,100,1],[2,3,4],[10000,6,7],[5000,90,100]])
print(a.shape) # (10, 2)
print(c.shape) # (7,  3)

print("=====a=====")
b = outliers(a)
print("이상치의 위치 :", b)
print("=====a=====")

print("=====c=====")
b2 = outliers(c)
print("이상치의 위치 :", b2)
print("=====c=====")

''' c 데이터 배열

1    , 2  , 3
3    , 4  , 10000
6    , 7  , 5000
90   ,100 , 1
2    , 3  , 4
10000,6   , 7
5000 ,90  , 100

'''
