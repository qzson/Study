# 20-06-03_17
# 시험 리뷰

# ==== csv 데이터 불러오기
import numpy as np
import pandas as pd

samsung = pd.read_csv('./data/csv/samsung.csv',
                      index_col = 0,        # '일자'는 데이터가 아니다. 따라서, 첫 열을 인덱스로 잡는다.
                      header = 0,           # '시가 ~ 거래량'도 데이터가 X, 따라서, 첫 행을 헤더로 잡는다.
                                            # 둘다 None 으로 잡으면, 위의 '값'들이 데이터로 잡힌다.
                      sep = ',',
                      encoding = 'CP949')

hite = pd.read_csv('./data/csv/hite.csv',
                   index_col = 0,           
                   header = 0,
                   sep = ',',
                   encoding = 'CP949')

# print(samsung.head())
# print(hite.head())
print(samsung.shape)    # (700, 1)
print(hite.shape)       # (720, 5)
                        # 함정이 3개나 있었다. (hite 시가 외에 NaN 값 4개 와, 509~700,720행 의 NaN 값들)

# ==== NaN 제거하기
# 방법 1 (fillna, dorpna 함수 사용)
samsung = samsung.dropna(axis = 0)
                        # 0 = 행 / 1 = 열 / 기입 값에 따라 제거 진행
# print(samsung)
print(samsung.shape)    # (509, 1)

hite = hite.fillna(method = 'bfill')
                            # bfill = 전날 행으로 채우겠다. / fill = 다음 행으로 채우겠다.
                            # bfill로 한 이유는 아래 값들을 보면, 그냥 서로 비슷하니까
                            # 결론 >>> 결측치 NaN 부분에 값을 주기 위해서
hite = hite.dropna(axis = 0)
                  # 위에 fillna를 적용 시켜서 NaN 제거할 행이 없지만 NaN이 있었다면, 그 해당 행은 제거 된다.
# print(hite)
''' print(hite) 방법 1
              시가    고가     저가    종가    거래량
 일자
 2020-06-02  39,000  38,750  36,000  38,750  1,407,345
 2020-06-01  36,000  38,750  36,000  38,750  1,407,345
 '''

# 방법 2
# hite = hite[0:509]
# hite.iloc[0, 1:5] = [10,20,30,40]
    # iloc (index location) : [안에 숫자로 들어간다. 0행의 1부터 5까지] = [이후는 넣을 값]
# hite.loc['2020-06-02', '고가':'거래량'] = ['10','20','30','40']
    # loc                   : 인덱스나 헤더 기준으로 기입하여 바로 수정할 때는 loc 사용
# print(hite)
print(hite.shape) # (509, 5)
''' print(hite) 방법 2
              시가    고가     저가    종가     거래량
 일자
 2020-06-02  39,000      10      20      30         40
 2020-06-01  36,000  38,750  36,000  38,750  1,407,345
 '''

# 또 다른 방법 (자습)
 # 결측치를 y_predict 잡고 그 아래 값들을 x_train, test 모델 짜서 예측하는 방법도 있다.
 # 이런 경우 xgbooster나 randomforest 의 강력한 것을 사용하지만, 이건 간단하기 때문에 dnn 정도로 해도 될 것 같다.

# ==== 데이터 int 형변환
for i in range(len(samsung.index)):
       samsung.iloc[i,0] = int(samsung.iloc[i,0].replace(',', ''))
# print(samsung) # int 변환된 samsung data

for i in range(len(hite.index)):
       for j in range(len(hite.iloc[i])):
              hite.iloc[i,j] = int(hite.iloc[i,j].replace(',', ''))
# print(hite)    # int 변환된 samsung data
print(type(samsung.iloc[i,0])) # <class 'int'>
print(type(hite.iloc[i,j])) # <class 'int'>

# ==== 데이터 인덱스 정렬 (오름차순으로)
samsung_sort = samsung.sort_values(['일자'], ascending=[True])
hite_sort = hite.sort_values(['일자'], ascending=[True])
# print(samsung_sort)
# print(hite_sort)

# ==== 데이터 numpy 변환 및 저장
samsung_npy = samsung_sort.values
hite_npy = hite_sort.values
print(type(samsung_npy), type(hite_npy)) # 둘다 NUMPY
print(samsung_npy.shape, hite_npy.shape) # 삼성 (509,1) / 하이트 (509, 5)

# np.save
np.save('./data/samsung_test.npy', arr=samsung_npy)
np.save('./data/hite_test.npy', arr=hite_npy)
