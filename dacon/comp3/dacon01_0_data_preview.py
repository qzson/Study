# 20-06-09
# Dacon : 진동데이터 활용 충돌체 탐지
# 진동데이터로 충돌체의 x좌표, y좌표, 질량, 속도 예측

'''
- 니즈
    : 시간과 가속도 데이터 활용, 충돌체 매개변수 (X,Y,M,V) 예측
    : 즉, [x좌표, y좌표, 질량, 속도] 예측

_features.csv
    : 가속도 센서로 측정한 충격신호 (시간, 가속도 데이터)

train_target.csv
    : 충돌체 매개변수

sample_submission.csv
    : 제출 양식

train_features 컬럼 구성 (= test_features)
[id, Time, S1, S2, S3, S4]
- id        : 충돌체 고유 아이디
- Time      : 관측시간
- S1 ~ S4   : 각 센서에서 측정된 가속도 (mm/s^2)

train_target 컬럼 구성 (= sample_submission)
[id, X, Y, M, V]
- id        : 충돌체 고유 아이디
- X, Y      : 충격하중이 가해진 X, Y 좌표 (mm)
- M         : 충돌체의 질량 (g)
- V         : 충돌체의 충돌 속도 (m/s)

'''
# 평가 지표 (MSPE = MSE)
import numpy as np

def kaeri_metric(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: KAERI metric
    '''
    
    return 0.5 * E1(y_true, y_pred) + 0.5 * E2(y_true, y_pred)


### E1과 E2는 아래에 정의됨 ###

def E1(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: distance error normalized with 2e+04
    '''
    
    _t, _p = np.array(y_true)[:,:2], np.array(y_pred)[:,:2]
    
    return np.mean(np.sum(np.square(_t - _p), axis = 1) / 2e+04)


def E2(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: sum of mass and velocity's mean squared percentage error
    '''
    
    _t, _p = np.array(y_true)[:,2:], np.array(y_pred)[:,2:]
    
    
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06)), axis = 1))


### 개인 정리 ###

# X,Y,M,V 의 변수로 총 2800회의 시물레이션 수행
# 시간 - 가속도 형태의 시계열 데이터를 추출
# 데이터가 2800개 세트로 구성
# Training Dataset : 2800개 시계열 데이터

# 충격 위치 (x, y)
#   : -400 ~ 400 mm, (x, y = 100 mm)
# 강구 질량 (m)
#   : 25 ~ 175 g,    (g, m = 25 g)
# 충돌 속도 (v)
#   : 0.2 ~ 1.0 m/s, (v    = 0.2 m/s)
