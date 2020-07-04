import numpy as np
import pandas as pd

data = pd.read_csv('./data/dacon/comp4/201901-202003.csv')
# submission = pd.read_csv('./data/dacon/comp4/submission.csv', index_col=0)

# print(data.head())
# print(data.tail())
#           REG_YYMM    CARD_SIDO_NM CARD_CCG_NM STD_CLSS_NM HOM_SIDO_NM HOM_CCG_NM  AGE  SEX_CTGO_CD  FLC  CSTMR_CNT      AMT  CNT
# 0           201901    강원         강릉시  건강보조식품 소매업          강원        강릉시  20s            1    1          4   311200    4
# 1           201901    강원         강릉시  건강보조식품 소매업          강원        강릉시  30s            1    2          7  1374500    8
# 2           201901    강원         강릉시  건강보조식품 소매업          강원        강릉시  30s            2    2          6   818700    6
# 3           201901    강원         강릉시  건강보조식품 소매업          강원        강릉시  40s            1    3          4  1717000    5
# 4           201901    강원         강릉시  건강보조식품 소매업          강원        강릉시  40s            1    4          3  1047300    3
#           REG_YYMM    CARD_SIDO_NM CARD_CCG_NM STD_CLSS_NM HOM_SIDO_NM HOM_CCG_NM  AGE  SEX_CTGO_CD  FLC  CSTMR_CNT     AMT  CNT
# 24697787    202003    충북         충주시    휴양콘도 운영업          충북        충주시  30s            1    2          3   43300    4
# 24697788    202003    충북         충주시    휴양콘도 운영업          충북        충주시  40s            1    3          3   35000    3
# 24697789    202003    충북         충주시    휴양콘도 운영업          충북        충주시  50s            1    4          4  188000    6
# 24697790    202003    충북         충주시    휴양콘도 운영업          충북        충주시  50s            2    4          4   99000    6
# 24697791    202003    충북         충주시    휴양콘도 운영업          충북        충주시  60s            1    5          3  194000    3

# print(submission.head())
# print(submission.tail())
#     REG_YYMM   CARD_SIDO_NM           STD_CLSS_NM          AMT
# id
# 0     202004           강원            건강보조식품 소매업    0       
# 1     202004           강원               골프장 운영업    0
# 2     202004           강원           과실 및 채소 소매업    0        
# 3     202004           강원     관광 민예품 및 선물용품 소매업    0   
# 4     202004           강원  그외 기타 분류안된 오락관련 서비스업    0
#       REG_YYMM CARD_SIDO_NM            STD_CLSS_NM  AMT
# id
# 1389    202007           충북  피자 햄버거 샌드위치 및 유사 음식점업    0
# 1390    202007           충북                한식 음식점업    0
# 1391    202007           충북                    호텔업    0
# 1392    202007           충북          화장품 및 방향제 소매업    0
# 1393    202007           충북               휴양콘도 운영업    0

data = data.fillna('')

df = data.copy()
df = df[['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM', 'AMT']]
df = df.groupby(['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM']).sum().reset_index(drop=False)
df = df.loc[df['REG_YYMM']==202003]
df = df[['CARD_SIDO_NM', 'STD_CLSS_NM', 'AMT']]

# print(df.head())
# print(df.tail())
#      CARD_SIDO_NM        STD_CLSS_NM         AMT
# 8829           강원         건강보조식품 소매업    96059012    
# 8830           강원            골프장 운영업  2915797995       
# 8831           강원        과실 및 채소 소매업   994816943     
# 8832           강원  관광 민예품 및 선물용품 소매업    13317300
# 8833           강원    그외 기타 스포츠시설 운영업     2075000 

#      CARD_SIDO_NM            STD_CLSS_NM          AMT
# 9433           충북  피자 햄버거 샌드위치 및 유사 음식점업   1315245299
# 9434           충북                한식 음식점업  16152482704
# 9435           충북                    호텔업     15248550
# 9436           충북          화장품 및 방향제 소매업    428881434      
# 9437           충북               휴양콘도 운영업     12733490
print(df.shape)

submission = pd.read_csv('./data/dacon/comp4/submission.csv', index_col=0)
submission = submission.loc[submission['REG_YYMM']==202004]
submission = submission[['CARD_SIDO_NM', 'STD_CLSS_NM']]
submission = submission.merge(df, left_on=['CARD_SIDO_NM', 'STD_CLSS_NM'], right_on=['CARD_SIDO_NM', 'STD_CLSS_NM'], how='left')
submission = submission.fillna(0)
AMT = list(submission['AMT'])*2

submission = pd.read_csv('./data/dacon/comp4/submission.csv', index_col=0)
submission['AMT'] = AMT
submission.to_csv('./data/dacon/comp4/submission2.csv', encoding='utf-8-sig')
print(submission.head())
print(submission.tail())
print(submission.shape)