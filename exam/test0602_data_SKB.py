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

import numpy as np
import pandas as pd

# 삼성, 하이트 판다스 불러오기
samsung_pd = pd.read_csv('./data/csv/samsung.csv', index_col = 0, header = 0, encoding='CP949', sep = ',')
hite_pd = pd.read_csv('./data/csv/hite.csv', index_col = 0, header = 0,  encoding='CP949', sep = ',')
print(samsung_pd)
print(hite_pd)
 # '일자' 컬럼은 데이터가 아니므로 index_col = 0, 설정 // 첫 행은 이름표 이므로 header = 0, 으로 설정
 # encoding='cp949' 를 써야 작동 (한글형 헤더라서? 받아들일 )
 # 프린트로 끝값을 찾을 수 없음 그러므로 head를 이용해보자.

print()

print(samsungdt.shape)
print(hitedt.shape)

# ssdt = samsungdt[1:509]
# # print(ssdt) # str
# htdt = hitedt[1:509]
# # print(htdt) # str

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
'''