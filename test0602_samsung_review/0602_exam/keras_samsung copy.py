# 20-06-02 화 시험
# 06-03 아침 삼성전자 시가 맞추기

''' 시험 조건
1. 6/3 삼성전자 시가 맞추기
2. CSV 데이터는 건들지 말 것
3. 앙상블 모델 사용
   hite + samsung
4. 6시 mail 발송 제목은 "손기범 [0602시험] 57000원"
   첨부   : 소스, npy, h5
   소스명 : test0602_SKB.py
'''
# 날짜 순이 거꾸로 되어있다. 알아서 처리


# 1. 데이터

# data save
import numpy as np
import pandas as pd

samsungdt = pd.read_csv('./data/csv/samsung.csv', index_col = 0, encoding='CP949', header = 0, sep = ',')
# print(samsungdt.head(509))
# print(samsungdt.shape)

hitedt = pd.read_csv('./data/csv/hite.csv', index_col = 0, encoding='CP949', header = 0, sep = ',')
# print(hitedt.head(509))
# print(hitedt.shape)

ssdt = samsungdt[1:509]
# print(ssdt) # str
htdt = hitedt[1:509]
# print(htdt) # str

'''
# samsung
                시가
일자
2020-06-02  51,000
2020-06-01  50,800
2020-05-29  50,000
2020-05-28  51,100
2020-05-27  48,950
...            ...
2018-05-11  52,000
2018-05-10  51,700
2018-05-09  52,600
2018-05-08  52,600
2018-05-04  53,000

[509 rows x 1 columns]
(700, 1)


# hite
                시가      고가      저가      종가        거래량
일자
2020-06-02  39,000     NaN     NaN     NaN        NaN
2020-06-01  36,000  38,750  36,000  38,750  1,407,345
2020-05-29  35,900  36,750  35,900  36,000    576,566
2020-05-28  36,200  36,300  35,500  35,800    548,493
2020-05-27  35,900  36,450  35,800  36,400    373,464
...            ...     ...     ...     ...        ...
2018-05-11  21,000  21,550  21,000  21,400    229,912
2018-05-10  21,150  21,250  20,900  21,000    215,315
2018-05-09  21,050  21,200  20,950  21,100    165,195
2018-05-08  21,450  21,550  21,000  21,050    250,520
2018-05-04  21,400  21,600  21,350  21,550    123,592

[509 rows x 5 columns]
(720, 5)

'''
# 데이터 int 형 변환
for i in range(len(ssdt.index)):
       ssdt.iloc[i,0] = int(ssdt.iloc[i,0].replace(',', ''))
# print(ssdt) # int

for i in range(len(htdt.index)):
       for j in range(len(htdt.iloc[i])):
              htdt.iloc[i,j] = int(htdt.iloc[i,j].replace(',', ''))
# print(htdt) # int

# 정렬
ssdt = ssdt.sort_values(['일자'], ascending=[True])
htdt = htdt.sort_values(['일자'], ascending=[True])
print(ssdt)
print(htdt)

# npy 변환
ssdt = ssdt.values
htdt = htdt.values
print(type(ssdt), type(htdt)) # 둘다 NUMPY
print(ssdt.shape, htdt.shape) # 삼성 (508,1) / 하이트 (508, 5)
# np.save
# np.save('./data/samsung.npy', arr=ssdt)
# np.save('./data/hite.npy', arr=htdt)

# hite, samsung npy 데이터
print(htdt[0:507])

'''
# hite
            시가    고가   저가   종가     거래량
일자
2018-05-04  21400  21600  21350  21550   123592
2018-05-08  21450  21550  21000  21050   250520
2018-05-09  21050  21200  20950  21100   165195
2018-05-10  21150  21250  20900  21000   215315
2018-05-11  21000  21550  21000  21400   229912
...           ...    ...    ...    ...      ...
2020-05-26  36450  36500  35750  36100   409419
2020-05-27  35900  36450  35800  36400   373464
2020-05-28  36200  36300  35500  35800   548493
2020-05-29  35900  36750  35900  36000   576566
2020-06-01  36000  38750  36000  38750  1407345

[508 rows x 5 columns]

# samsung
            시가
일자
2018-05-04  53000
2018-05-08  52600
2018-05-09  52600
2018-05-10  51700
2018-05-11  52000
...           ...
2020-05-26  48700
2020-05-27  48950
2020-05-28  51100
2020-05-29  50000
2020-06-01  50800

[508 rows x 1 columns]

'''

def split_xy3(dataset, time_steps, y_column):
       x, y = list(), list()
   for i in range(len(dataset)):
      x_end_number = i + time_steps
      y_end_number = x_end_number + y_column

      if y_end_number > len(dataset):
         break
      tmp_x = dataset[i:x_end_number, :]
      tmp_y = dataset[x_end_number+1, 0]
      x.append(tmp_x)
      y.append(tmp_y)
   return np.array(x), np.array(y)

0~ 508
0 + 3
3 + 1

504 + 3
507 + 1
dt[504:507,:]
dt[508,0]


def split_xy3(dataset, time_steps, y_column):
       x, y = list(), list()
   for i in range(len(dataset)):
      x_end_number = i + time_steps
      y_end_number = x_end_number + y_column

      if y_end_number > len(dataset):
         break
      tmp_x = dataset[i:x_end_number, :]
      tmp_y = dataset[x_end_number:y_end_number, 0]
      x.append(tmp_x)
      y.append(tmp_y)
   return np.array(x), np.array(y)

504 + 3
507 + 1

504:507, :
507:508, 0