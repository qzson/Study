# 20-06-08_20
# mon / 16:00 ~

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from keras.utils import np_utils
from sklearn.pipeline import Pipeline, make_pipeline


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
print(test_npy.shape)

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

# x 전처리
sc = MinMaxScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)


''' 2. 모델 '''

inputs = Input(shape=(71,), name='input')
x = Dense(1024, activation='relu', name='hidden1')(inputs)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', name='hidden2')(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu', name='hidden3')(x)
x = Dropout(0.2)(x)
output = Dense(4, activation='relu', name='outputs')(x)

model = Model(inputs=inputs, outputs=output)

model.summary()


''' 3. 훈련 '''
model.compile(loss='mae', optimizer='adam', metrics=['mae'])  
model.fit(x_train, y_train, epochs=200, batch_size=32, validation_split=0.25, verbose=1)


''' 4. 평가, 예측 '''
loss = model.evaluate(x_test, y_test, batch_size=32)
print("mae :", loss)

x_pred = test
# y_pred = submission
x_pred = sc.transform(x_pred)

y_pred = model.predict(x_pred)
print(y_pred)

# # submit 열어보면 왼쪽에 인덱스 위쪽에 헤더
# y_pred.to_csv('./dacon/y_predict.csv')
#pred할 submit 파일을 만든다

a = np.arange(10000, 20000)
y_pred = pd.DataFrame(y_pred, a)
y_pred.to_csv('./data/dacon/comp1/sample_submission.csv', index = True, header=['hhb','hbo2','ca','na'],index_label='id')
