# 20-06-08_20
# mon / 16:00 ~

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


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

y_train1 = y_train[:, 0]
y_train2 = y_train[:, 1]
y_train3 = y_train[:, 2]
y_train4 = y_train[:, 3]

y_test1 = y_test[:, 0]
y_test2 = y_test[:, 1]
y_test3 = y_test[:, 2]
y_test4 = y_test[:, 3]
print(y_test1.shape)


''' 2. 모델 '''
model = GradientBoostingRegressor()
# model = XGBRegressor()

''' 3. GB, XGB 훈련 평가 결과 출력 '''
model.fit(x_train, y_train1)
score1 = model.score(x_test, y_test1)
print("score1 : ", score1)
print(model.feature_importances_)
y_pred1 = model.predict(x_pred)

model.fit(x_train, y_train2)
score2 = model.score(x_test, y_test2)
print("score2 : ", score2)
print(model.feature_importances_)
y_pred2 = model.predict(x_pred)

model.fit(x_train, y_train3)
score3 = model.score(x_test, y_test3)
print("score3 : ", score3)
print(model.feature_importances_)
y_pred3 = model.predict(x_pred)

model.fit(x_train, y_train4)
score4 = model.score(x_test, y_test4)
print("score4 : ", score4)
print(model.feature_importances_)
y_pred4 = model.predict(x_pred)

print(y_pred1.shape)
print(y_pred2.shape)
print(y_pred3.shape)
print(y_pred4.shape)

sub = pd.DataFrame({
    'id': np.array(range(10000, 20000)),
    'hhb': y_pred1,
    'hbo2': y_pred2,
    'ca': y_pred3,
    'na': y_pred4
})
print(sub)

''' 4. 저장 '''
# sub.to_csv('./data/dacon/comp1/sample_submission6.csv', index=False)


''' 5. matplot 출력 '''
def plot_feature_importances(model):
    plt.figure(figsize= (10, 40))
    n_features = x_data.shape[1]                                                # n_features : column 개수
    plt.barh(np.arange(n_features), model.feature_importances_, align='center') # barh       : 가로방향 bar chart
    plt.yticks(np.arange(n_features), x_data.columns)                           # tick       : 축상의 위치표시 지점
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)                                                    # Y축 최소, 최대값 설정 / X축은 xlim 사용

plot_feature_importances(model)
plt.show()

## XGB 결과 ##
# score1 :  0.6615769100566586
# score2 :  0.1195446469991307
# score3 :  0.16822174177705862
# score4 :  0.0796433038534704

## GB 결과 ##
# score1 :  -0.01713487274786729
# score2 :  -0.010405176062405319
# score3 :  -0.011401902417798926
# score4 :  -0.009811167850817037
