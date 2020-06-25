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

# # y_train1 = y_train[:, 0]
# # y_train2 = y_train[:, 1]
# # y_train3 = y_train[:, 2]
# # y_train4 = y_train[:, 3]

# # y_test1 = y_test[:, 0]
# # y_test2 = y_test[:, 1]
# # y_test3 = y_test[:, 2]
# # y_test4 = y_test[:, 3]

xgbr = XGBRegressor()
# model.fit(x_train, y_train)
# score = model.score(x_test, y_test)
# print('R2 :', score)

model = MultiOutputRegressor(xgbr)
model.fit(x_train,y_train)
# print(len(model.estimators_))
# print(model.estimators_[0].feature_importances_)

for i in range(len(model.estimators_)):
    threshold = np.sort(model.estimators_[i].feature_importances_)

    for thresh in threshold:
        selection = SelectFromModel(model.estimators_[i], threshold=thresh, prefit=True)

        param = {
            'n_estimators': [200],
            'learning_rate': [0.7],
            'max_depth': [10],
            'colsample_bytree': [0.7],
            'reg_alpha': [1],
            'scale_pos_weight': [1]
        }
        gridcv = RandomizedSearchCV(XGBRegressor(n_jobs = 6), param, cv=5, n_jobs = 6)
        
        select_x_train = selection.transform(x_train)
        selection_model = MultiOutputRegressor(gridcv)
        selection_model.fit(select_x_train, y_train)
        
        select_x_test = selection.transform(x_test)
        y_pred = selection_model.predict(select_x_test)
        mae = mean_absolute_error(y_test, y_pred)

        score = r2_score(y_test, y_pred)
        print('thresh=%.3f, n=%d, R2: %.2f%%, MAE: %.3f' %(thresh, select_x_train.shape[1], score*100.0, mae))
        # print(gridcv.best_params_)

        select_x_pred = selection.transform(x_pred)
        y_predict = selection_model.predict(select_x_pred)

        a = np.arange(10000,20000)
        submission = pd.DataFrame(y_predict, a)
        submission.to_csv('./data/dacon/comp1/kb_XG_%i_%.5f.csv' %(i, mae),index = True, header=['hhb','hbo2','ca','na'],index_label='id')
    


# score = model.score(x_test,y_test)
# print(score)
# y4 = model.predict(test.values)