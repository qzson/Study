# 20-06-09_21
# Dacon : 진동데이터 활용 충돌체 탐지
# ML 버전 // randomforestregressor + pipeline + randomizedseachCV


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# pandas.csv 불러오기
# train_x = pd.read_csv('./data/dacon/comp3/train_features.csv', header=0, index_col=0)
# train_y = pd.read_csv('./data/dacon/comp3/train_target.csv', header=0, index_col=0)
# test_x = pd.read_csv('./data/dacon/comp3/test_features.csv', header=0, index_col=0)


from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt

### 데이터 ###
x = pd.read_csv('./data/dacon/comp3/x_train2.csv',
                      index_col = 0, header = 0,
                      encoding = 'utf-8')
print(x.head())

y = pd.read_csv('./data/dacon/comp3/train_target.csv',
                index_col = 0, header = 0,
                encoding = 'utf-8')

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state=66, shuffle=True)
print(x_train.shape)        # (2240, 4)
print(x_test.shape)         # (560, 4)
print(y_train.shape)        # (2240, 4)
print(y_test.shape)         # (560, 4)

x_train = x_train.values
x_test = x_test.values
y_train = y_train.values
y_test = y_test.values

x_pred = pd.read_csv('./data/dacon/comp3/x_pred.csv',
                     index_col = 0, header = 0)
print(x_pred.head())

x_pred = x_pred.values
print(x_pred.shape)         # (700, 4)



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