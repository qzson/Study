# 20-06-11_23

# 데이콘 1 생체 / XGB 적용 (PCA 통해서 y타겟값 쉐이프 변환)

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


''' 1. 데이터 '''
train = pd.read_csv('./data/dacon/comp1/train.csv', header=0, index_col=0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header=0, index_col=0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header=0, index_col=0)

print('train.shape :', train.shape)             # (10000, 75) : x,y_train, test
print('test.shape :', test.shape)               # (10000, 71) : x_predict
print('submission.shape :', submission.shape)   # (10000,  4) : y_predict

## 약 2000개의 데이터 결측 확인
# train.isna().sum().plot()
# test.isna().sum().plot()
# plt.show()

## 결측치 제거
# print(train.isnull().sum())
train = train.interpolate()
test = test.interpolate()

train = train.fillna(train.mean())
test = test.fillna(test.mean())
print(train.head())
# print(train.tail())

# plt.figure(figsize=(4,12))
# sns.heatmap(train.corr().loc['rho':'990_dst', 'hhb':].abs())
# plt.show()

## 데이터 iloc 슬라이싱
x_data = train.iloc[:, :71]
y_data = train.iloc[:, 71:]
print(x_data.head())
print(y_data.head())

## 데이터 npy 형변환
x_npy = x_data.values
y_npy = y_data.values
test_npy = test.values
# print(type(train_npy))      # npy
# print(type(test_npy))       # npy
x_pred = test_npy

# x = train_npy[:, :-4]
# y = train_npy[:, -4:]
# print(x.shape)              # 10000, 71
# print(y.shape)              # 10000, 4

## 데이터 스플릿
x_train, x_test, y_train, y_test = train_test_split(x_npy, y_npy, test_size = 0.2, random_state = 66, shuffle = True)
# print(x_train.shape)        # (8000, 71)
# print(x_test.shape)         # (2000, 71)
# print(y_train.shape)        # (8000, 4)
# print(y_test.shape)         # (2000, 4)

sc = StandardScaler()
sc.fit(y_train)
y_train= sc.transform(y_train)
y_test= sc.transform(y_test)

pca = PCA(n_components=1)
pca.fit(y_train)
y_train = pca.transform(y_train)
y_test = pca.transform(y_test)
print(y_test.shape)

model = XGBRegressor()

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print(model.feature_importances_)
print(score)
y_pred = model.predict(x_pred)
# print(y_pred)
y_pred1 = model.predict(x_test)

print('mae: ', mean_absolute_error(y_test, y_pred1))
# mae:  0.6744712265574293

# PCA 복원 전 차원 정립 (백터형태로 나오기 때문)
print(y_pred.shape) # (10000,)
y_pred = y_pred.reshape(10000, 1)

# PCA 인버스 (복원)
y_pred = pca.inverse_transform(y_pred)
y_pred = sc.inverse_transform(y_pred)
print(y_pred.shape) # (10000, 4)
print(y_pred)

a = np.arange(10000, 20000)
y_pred = pd.DataFrame(y_pred, a)
y_pred.to_csv('./data/dacon/comp1/sample_submission8.csv', index = True, header=['hhb','hbo2','ca','na'], index_label='id')