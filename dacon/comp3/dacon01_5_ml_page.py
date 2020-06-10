# 데이콘 2 데이터 정제

import pandas as pd
import numpy as np

# 데이터 불러오기
train_features = pd.read_csv('./data/dacon/comp3/train_features.csv')
train_target = pd.read_csv('./data/dacon/comp3/train_target.csv', index_col = 'id')
test_features = pd.read_csv('./data/dacon/comp3/test_features.csv')

# 데이터 형태 확인
print(f'train_features {train_features.shape}')
print(f'train_target {train_target.shape}')
print(f'test_features {test_features.shape}')
# train_features (1050000, 6)
# train_target (2800, 4)
# test_features (262500, 6)

def preprocessing_KAERI(data) :
    '''
    data: train_features.csv or test_features.csv
    
    return: Random Forest 모델 입력용 데이터
    '''
    
    # 충돌체 별로 0.000116 초 까지의 가속도 데이터만 활용해보기 
    _data = data.groupby('id').head(30)
    
    # string 형태로 변환
    _data['Time'] = _data['Time'].astype('str')
    
    # Random Forest 모델에 입력 할 수 있는 1차원 형태로 가속도 데이터 변환
    _data = _data.pivot_table(index = 'id', columns = 'Time', values = ['S1', 'S2', 'S3', 'S4'])
    
    # column 명 변환
    _data.columns = ['_'.join(col) for col in _data.columns.values]
    
    return _data
train_features = preprocessing_KAERI(train_features)
test_features = preprocessing_KAERI(test_features)
print(f'train_features {train_features.shape}')
print(f'test_features {test_features.shape}')
# train_features (2800, 120)
# test_features (700, 120)

print(train_features)