# 200621 19일 수업 데이콘 적용


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error



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
print(train.tail())

# plt.figure(figsize=(4,12))
# sns.heatmap(train.corr().loc['rho':'990_dst', 'hhb':].abs())
# plt.show()

x_data = train.iloc[:, :71]
y_data = train.iloc[:, 71:]
print(x_data.head())
print(y_data.head())

x_npy = x_data.values
y_npy = y_data.values
test_npy = test.values
# # print(type(train_npy))      # npy
# # print(type(test_npy))       # npy
x_pred = test_npy

# 스플릿
x_train, x_test, y_train, y_test = train_test_split(x_npy, y_npy, test_size = 0.2, random_state = 66, shuffle = True)
print(x_train.shape)        # (8000, 71)
print(x_test.shape)         # (2000, 71)
print(y_train.shape)        # (8000, 4)
print(y_test.shape)         # (2000, 4)

y_train1 = y_train[:, 0]
y_train2 = y_train[:, 1]
y_train3 = y_train[:, 2]
y_train4 = y_train[:, 3]

y_test1 = y_test[:, 0]
y_test2 = y_test[:, 1]
y_test3 = y_test[:, 2]
y_test4 = y_test[:, 3]
print(y_train1.shape)

model = XGBRegressor(n_estimators=550,
                    learning_rate=0.05,
                    max_depth=8,
                    colsample_bytree=0.7,
                    reg_alpha=1,
                    scale_pos_weight=1,
                    reg_lambda = 1.1,
                    n_jobs=6)


model.fit(x_train, y_train1, verbose=False, eval_metric=['logloss','mae'],
                eval_set=[(x_train, y_train1), (x_test, y_test1)],
                early_stopping_rounds=20)
score1 = model.score(x_test, y_test1)
print("score1 : %.4f" %(score1 * 100.0))
# print(model.feature_importances_)
y_pred_1 = model.predict(x_test)
mae1 = mean_absolute_error(y_test1, y_pred_1)
print('mae1 : %.4f' %(mae1))
y_pred1 = model.predict(x_pred)


model.fit(x_train, y_train2, verbose=False, eval_metric=['logloss','mae'],
                eval_set=[(x_train, y_train2), (x_test, y_test2)],
                early_stopping_rounds=20)
score2 = model.score(x_test, y_test2)
print("score2 : %.4f" %(score2 * 100.0))
# print(model.feature_importances_)
y_pred_2 = model.predict(x_test)
mae2 = mean_absolute_error(y_test2, y_pred_2)
print('mae2 : %.4f' %(mae2))
y_pred2 = model.predict(x_pred)

model.fit(x_train, y_train3, verbose=False, eval_metric=['logloss','mae'],
                eval_set=[(x_train, y_train3), (x_test, y_test3)],
                early_stopping_rounds=20)
score3 = model.score(x_test, y_test3)
print("score3 : %.4f" %(score3 * 100.0))
# print(model.feature_importances_)
y_pred_3 = model.predict(x_test)
mae3 = mean_absolute_error(y_test3, y_pred_3)
print('mae3 : %.4f' %(mae3))
y_pred3 = model.predict(x_pred)

model.fit(x_train, y_train4, verbose=False, eval_metric=['logloss','mae'],
                eval_set=[(x_train, y_train4), (x_test, y_test4)],
                early_stopping_rounds=20)
score4 = model.score(x_test, y_test4)
print("score4 : %.4f" %(score4 * 100.0))
# print(model.feature_importances_)
y_pred_4 = model.predict(x_test)
mae4 = mean_absolute_error(y_test4, y_pred_4)
print('mae4 : %.4f' %(mae4))
y_pred4 = model.predict(x_pred)

# print(y_pred1.shape)
# print(y_pred1[0])
# print(y_pred2.shape)
# print(y_pred2[0])
# print(y_pred3.shape)
# print(y_pred4.shape)

sub = pd.DataFrame({
    'id': np.array(range(10000, 20000)),
    'hhb': y_pred1,
    'hbo2': y_pred2,
    'ca': y_pred3,
    'na': y_pred4
})
print(sub)

''' 4. 저장 '''
sub.to_csv('./data/dacon/comp1/sub_xgb_01.csv', index=False)

# learning_rate = 0.05
# max_depth = 10
# score1 : 69.9365
# mae1 : 1.2077
# score2 : 19.0119
# mae2 : 0.7242
# score3 : 22.6384
# mae3 : 2.1385
# score4 : 12.1703
# mae4 : 1.4083

# learning_rate = 0.05
# max_depth = 8
# score1 : 70.3881
# mae1 : 1.2070
# score2 : 19.8141
# mae2 : 0.7209
# score3 : 23.3364
# mae3 : 2.1256
# score4 : 12.8546
# mae4 : 1.4104

# learning_rate = 0.05
# max_depth = 8
# colsample_bytree = 0.7
# score1 : 70.1660
# mae1 : 1.2076
# score2 : 21.2997
# mae2 : 0.7160
# score3 : 24.4317
# mae3 : 2.1133
# score4 : 15.1926
# mae4 : 1.3859

# n_estimators = 400
# learning_rate = 0.05
# max_depth = 8
# colsample_bytree = 0.7
# score1 : 70.8534
# mae1 : 1.1930
# score2 : 21.4974
# mae2 : 0.7150
# score3 : 24.8898
# mae3 : 2.1017
# score4 : 15.4050
# mae4 : 1.3812

# n_estimators = 500
# learning_rate = 0.05
# max_depth = 8
# colsample_bytree = 0.7
# score1 : 70.9467
# mae1 : 1.1907
# score2 : 21.4974
# mae2 : 0.7150
# score3 : 24.8898
# mae3 : 2.1017
# score4 : 15.4050
# mae4 : 1.3812

# n_estimators = 500
# learning_rate = 0.05
# max_depth = 8
# colsample_bytree = 0.7
# reg_lambda = 1.1
# score1 : 71.3501
# mae1 : 1.1817
# score2 : 21.3879
# mae2 : 0.7141
# score3 : 24.6554
# mae3 : 2.1047
# score4 : 15.2988
# mae4 : 1.3872

# n_estimators = 500
# learning_rate = 0.05
# max_depth = 8
# colsample_bytree = 0.7
# reg_lambda = 3
# score1 : 71.7730
# mae1 : 1.1769
# score2 : 20.7746
# mae2 : 0.7181
# score3 : 25.1594
# mae3 : 2.1079
# score4 : 14.6432
# mae4 : 1.3967