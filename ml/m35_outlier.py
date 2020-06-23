# 200623_26
# 이상치 제거

'''
1,2,3,100만,200만,6,7

1. 삭제 (문제가 있음- 평균시 달라지는..)
2. NaN 처리 후 보간법, fillna, bfillna 사용
3. 감성적인 데이터 분석 - 맞을 수도 있지만, 틀릴 수도 있다.
4. I Q R (전체 범위를 4등분 / 25% , 75% 부분에 1.5를 곱해서 범위를 벗어나는 애는 이상치로 판단한다)
'''

import numpy as np

def outliers(data_out):
    quartile_1, quartile_3 = np.percentile(data_out, [25, 75])
    print("1사 분위 :", quartile_1)
    print("3사 분위 :", quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

a = np.array([1,2,3,4,10000,6,7,5000,90,100])
# (10,)

b = outliers(a)
print("이상치의 위치 :", b)

# 실습   : 행렬을 입력해서 컬럼별로 이상치 발견하는 함수를 구현하기
# 파일명 : m36_outliers.py