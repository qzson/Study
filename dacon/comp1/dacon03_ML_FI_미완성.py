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
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


''' 1. 데이터 '''

train = pd.read_csv('./data/dacon/comp1/train.csv', header=0, index_col=0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header=0, index_col=0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header=0, index_col=0)

print('train.shape :', train.shape)             # (10000, 75) : x,y_train, test
print('test.shape :', test.shape)               # (10000, 71) : x_predict
print('submission.shape :', submission.shape)   # (10000,  4) : y_predict

# 약 2000개의 데이터 결측 확인
# train.isna().sum().plot()
# test.isna().sum().plot()
# plt.show()

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

train_npy = train.values
test_npy = test.values
# print(type(train_npy))      # npy
# print(type(test_npy))       # npy
x_pred = test_npy

x = train_npy[:, :-4]
y = train_npy[:, -4:]
# print(x.shape)              # 10000, 71
# print(y.shape)              # 10000, 4

# 스플릿
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 66, shuffle = True)
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


''' 2. 모델 '''
# # model = DecisionTreeRegressor(max_depth =4)                     # max_depth 몇 이상 올라가면 구분 잘 못함
# # model = RandomForestRegressor(n_estimators = 200, max_depth=3)
# # model = GradientBoostingRegressor()
model = XGBRegressor()

''' 3. GB, XGB 훈련 평가 결과 출력 '''
model.fit(x_train, y_train1)
score1 = model.score(x_test, y_test1)
print("score1 : ", score1)
# print(model.feature_importances_)
y_pred1 = model.predict(test)

model.fit(x_train, y_train2)
score2 = model.score(x_test, y_test2)
print("score2 : ", score2)
# print(model.feature_importances_)
y_pred2 = model.predict(test)

model.fit(x_train, y_train3)
score3 = model.score(x_test, y_test3)
print("score3 : ", score3)
# print(model.feature_importances_)
y_pred3 = model.predict(test)

model.fit(x_train, y_train4)
score4 = model.score(x_test, y_test4)
print("score4 : ", score4)
# print(model.feature_importances_)
y_pred4 = model.predict(test)

print(model.feature_importances_)

''' 3. 그 외 모델 훈련 '''
# model.fit(x_train, y_train)

''' 4. 평가 '''
# score = model.score(x_test, y_test)
# print(model.feature_importances_)
# print(score)

# ''' 5. 출력 '''
# def plot_feature_importances_cancer(model):
#     plt.figure(figsize= (10, 40))
#     n_features = x_train.shape[1]                                                # n_features : column 개수
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center') # barh       : 가로방향 bar chart
#     plt.yticks(np.arange(n_features), x_train.columns)                           # tick       : 축상의 위치표시 지점
#     plt.xlabel("Feature Importances")
#     plt.ylabel("Features")
#     plt.ylim(-1, n_features)                                                    # Y축 최소, 최대값 설정 / X축은 xlim 사용

# plot_feature_importances_cancer(model)
# plt.show()

# ''' 3. 훈련 '''
# model.fit(x_train, y_train)


# ''' 4. 평가, 예측 '''
# score = model.score(x_test, y_test)

# print('최적의 매개변수 :', model.best_params_)
# print('score :', score)


# y_pred = model.predict(x_pred)
# print(y_pred)

# a = np.arange(10000, 20000)
# y_pred = pd.DataFrame(y_pred, a)
# y_pred.to_csv('./data/dacon/comp1/sample_submission2.csv', index = True, header=['hhb','hbo2','ca','na'], index_label='id')
