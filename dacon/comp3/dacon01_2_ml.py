# 20-06-09_21
# Dacon : 진동데이터 활용 충돌체 탐지
# ML 버전 // randomforestregressor + pipeline + randomizedseachCV


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# pandas.csv 불러오기
train_x = pd.read_csv('./data/dacon/comp3/train_features.csv', header=0, index_col=0)
train_y = pd.read_csv('./data/dacon/comp3/train_target.csv', header=0, index_col=0)
test_x = pd.read_csv('./data/dacon/comp3/test_features.csv', header=0, index_col=0)


from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor


''' 1. 데이터 '''
# 초기 shape
# print(train_x.shape)    # (1050000, 5)
# print(train_y.shape)    # (2800, 4)
# print(test_x.shape)     # (262500, 5)

# pandas 데이터셋 컷
x = train_x.iloc[:, -4:]
y = train_y
x_pred = test_x.iloc[:, -4:]
# print(x.shape)          # (1050000, 4)
# print(y.shape)          # (2800, 4)
# print(x_pred.shape)     # (262500, 4)

# npy 형변환
x = x.values
y = y.values
x_pred = x_pred.values

# 2차원 reshape
x = x.reshape(2800, 375*4)
x_pred = x_pred.reshape(700, 375*4)
print(x.shape)      # (2800, 1500)
print(y.shape)      # (2800, 4)
print(x_pred.shape) # (700, 1500)

# 스플릿
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True)
print(x_train.shape)    # (2240, 1500)
print(x_test.shape)     # (560, 1500)
print(y_train.shape)    # (2240, 4)
print(y_test.shape)     # (560, 4)

parameters ={
    'rf__n_estimators' : [100],
    'rf__max_depth' : [10],
    'rf__min_samples_leaf' : [3],
    'rf__min_samples_split' : [5]
}


''' 2. 모델 '''
pipe = Pipeline([('scaler', MinMaxScaler()), ('rf', RandomForestRegressor())])
kfold = KFold(n_splits=5, shuffle=True)
model = RandomizedSearchCV(pipe, parameters, cv=kfold, n_jobs=-1)


''' 3. 훈련 '''
model.fit(x_train, y_train)


''' 4. 평가, 예측 '''
score = model.score(x_test, y_test)

print('최적의 매개변수 :', model.best_params_)
print('score :', score)


y_pred = model.predict(x_pred)
# print(y_pred)
y_pred1 = model.predict(x_test)


def kaeri_metric(y_test, y_pred1):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: KAERI metric
    '''
    
    return 0.5 * E1(y_test, y_pred1) + 0.5 * E2(y_test, y_pred1)


### E1과 E2는 아래에 정의됨 ###

def E1(y_test, y_pred1):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: distance error normalized with 2e+04
    '''
    
    _t, _p = np.array(y_test)[:,:2], np.array(y_pred1)[:,:2]
    
    return np.mean(np.sum(np.square(_t - _p), axis = 1) / 2e+04)


def E2(y_test, y_pred1):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: sum of mass and velocity's mean squared percentage error
    '''
    
    _t, _p = np.array(y_test)[:,2:], np.array(y_pred1)[:,2:]
    
    
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06)), axis = 1))

print(kaeri_metric(y_test, y_pred1))
print(E1(y_test, y_pred1))
print(E2(y_test, y_pred1))

a = np.arange(2800, 3500)
submission = pd.DataFrame(y_pred, a)
submission.to_csv('./dacon/comp3/comp3_sub3.csv', index = True, index_label= ['id'], header = ['X', 'Y', 'M', 'V'])