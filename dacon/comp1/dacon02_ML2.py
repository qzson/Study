# 20-06-08_20
# mon / 16:00 ~

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor


''' 1. 데이터 '''

train = pd.read_csv('./data/dacon/comp1/train.csv', header=0, index_col=0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header=0, index_col=0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header=0, index_col=0)

print('train.shape :', train.shape)             # (10000, 75) : x_train, test
print('test.shape :', test.shape)               # (10000, 71) : x_predict
print('submission.shape :', submission.shape)   # (10000,  4) : y_predict

'''
train.csv >> x_train,test / y_train,test 로 나누고
test.csv  >> x_pred
subm.csv  >> y_pred                      로 나눈다
'''

# 결측치가 어떤놈인지 함 보자 (컬럼당 얼마나 모여있는지)
print(train.isnull().sum())
train = train.interpolate()
test = test.interpolate()

train = train.fillna(method = 'bfill')
test = test.fillna(method = 'bfill')

train_npy = train.values
test_npy = test.values
# print(type(train_npy)) # npy
# print(type(test_npy)) # npy
print(test_npy.shape)   # (10000, 71)
x_pred = test_npy

x = train_npy[:, :-4]
y = train_npy[:, -4:]
# print(x.shape) # 10000, 71
# print(y.shape) # 10000, 4

# 스플릿
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 66, shuffle = True)
print(x_train.shape) # (8000, 71)
print(x_test.shape)  # (2000, 71)
print(y_train.shape) # (8000, 4)
print(y_test.shape)  # (2000, 4)


''' 2. 모델 '''
pipe = Pipeline([('scaler', RobustScaler()), ('rf', RandomForestRegressor())])


''' 3. 훈련 '''
pipe.fit(x_train, y_train)


''' 4. 평가, 예측 '''
score = pipe.score(x_test, y_test)

print('score :', score)


y_pred = pipe.predict(x_pred)
print(y_pred)

a = np.arange(10000, 20000)
y_pred = pd.DataFrame(y_pred, a)
y_pred.to_csv('./data/dacon/comp1/sample_submission5.csv', index = True, header=['hhb','hbo2','ca','na'], index_label='id')

# standardscaler (subm4)
# score : 0.28003031096471254
# 평가점수 1.49833