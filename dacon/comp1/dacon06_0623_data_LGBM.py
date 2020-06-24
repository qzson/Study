# 200623_26 수업내용 적용

# LGBM


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from xgboost import XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error



train = pd.read_csv('./data/dacon/comp1/train.csv', header=0, index_col=0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header=0, index_col=0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header=0, index_col=0)

print('train.shape :', train.shape)             # (10000, 75) : x,y_train, test
print('test.shape :', test.shape)               # (10000, 71) : x_predict
print('submission.shape :', submission.shape)   # (10000,  4) : y_predict
# print(train.head())

# how to edit numpy & pandas outliers
def outliers(data_out):
    out = []
    count = 0
    if str(type(data_out))== str("<class 'numpy.ndarray'>"):
        for col in range(data_out.shape[1]):
            data = data_out[:,col]
            print(data)

            quartile_1, quartile_3 = np.percentile(data,[25,75])
            print("1사분위 : ",quartile_1)
            print("3사분위 : ",quartile_3)
            iqr = quartile_3 - quartile_1
            lower_bound = quartile_1 - (iqr*1.5)
            upper_bound = quartile_3 + (iqr*1.5)
            out_col = np.where((data>upper_bound)|(data<lower_bound))
            print(out_col)
            data = data[out_col]
            print(f"{col+1}번째 행렬의 이상치 값: ", data)
            out.append(out_col)
            count += len(out_col)

    if str(type(data_out))== str("<class 'pandas.core.frame.DataFrame'>"):
        i=0
        print(data_out.columns)
        for col in data_out.columns:
            data = data_out[col].values
            # print(data)
            # print(type(data))
            quartile_1, quartile_3 = np.percentile(data,[25,75])
            # print("1사분위 : ",quartile_1)
            # print("3사분위 : ",quartile_3)
            iqr = quartile_3 - quartile_1
            lower_bound = quartile_1 - (iqr*1.5)
            upper_bound = quartile_3 + (iqr*1.5)
            out_col = np.where((data>upper_bound)|(data<lower_bound))
            # print('===out_col===')
            # print(out_col)
            # print('===out_col[0]===')
            # print(out_col[0], i)
            data_out.iloc[out_col[0],i]=np.nan
            data = data[out_col]
            # print(f"'{col}'의 이상치값: ", data)
            # print(type(data))
            i+=1
    return data_out

train = outliers(train)
test = outliers(test)

train = train.interpolate()
test = test.interpolate()

train = train.fillna(train.mean())
test = test.fillna(test.mean())

# print(f'train_head\n\n', train.head())
# print(f'test_head\n\n', test.head())
# print(f'train_tail\n\n', test.tail())
# print(f'test_tail\n\n', test.tail())

x_data = train.iloc[:, :71]
y_data = train.iloc[:, 71:]


x_npy = x_data.values
y_npy = y_data.values
x_pred = test.values

x_train, x_test, y_train, y_test = train_test_split(x_npy, y_npy, test_size = 0.2, random_state = 66, shuffle = True)
print(x_train.shape)        # (8000, 71)
print(x_test.shape)         # (2000, 71)
print(y_train.shape)        # (8000, 4)
print(y_test.shape)         # (2000, 4)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

y_train1 = y_train[:, 0]
y_train2 = y_train[:, 1]
y_train3 = y_train[:, 2]
y_train4 = y_train[:, 3]

y_test1 = y_test[:, 0]
y_test2 = y_test[:, 1]
y_test3 = y_test[:, 2]
y_test4 = y_test[:, 3]
print(y_train1.shape)

model = LGBMRegressor(n_estimators=550,
                      num_leaves=100,
                      max_depth=20,
                      min_child_samples=30,
                      learning_rate=0.04,
                      colsample_bytree=0.7,
                      reg_alpha=1,
                      reg_lambda = 1.1,
                      n_jobs=6)
# 100 / 20 / 30 / 0.05 / 0.5
# score1 : 72.3023
# mae1 : 1.1604
# score2 : 21.5578
# mae2 : 0.6963
# score3 : 25.3974
# mae3 : 2.0771
# score4 : 17.2373
# mae4 : 1.3498

# 100 / 20 / 30 / 0.04 / 0.7
# score1 : 72.2789
# mae1 : 1.1567
# score2 : 22.8805
# mae2 : 0.6918
# score3 : 26.0999
# mae3 : 2.0683
# score4 : 17.5051
# mae4 : 1.3463

model.fit(x_train, y_train1, verbose=False, eval_metric=['logloss'],
                eval_set=[(x_test, y_test1)],
                early_stopping_rounds=20)
score1 = model.score(x_test, y_test1)
print("score1 : %.4f" %(score1 * 100.0))
# print(model.feature_importances_)
y_pred_1 = model.predict(x_test)
mae1 = mean_absolute_error(y_test1, y_pred_1)
print('mae1 : %.4f' %(mae1))
y_pred1 = model.predict(x_pred)


model.fit(x_train, y_train2, verbose=False, eval_metric=['logloss'],
                eval_set=[(x_test, y_test2)],
                early_stopping_rounds=20)
score2 = model.score(x_test, y_test2)
print("score2 : %.4f" %(score2 * 100.0))
# print(model.feature_importances_)
y_pred_2 = model.predict(x_test)
mae2 = mean_absolute_error(y_test2, y_pred_2)
print('mae2 : %.4f' %(mae2))
y_pred2 = model.predict(x_pred)

model.fit(x_train, y_train3, verbose=False, eval_metric=['logloss'],
                eval_set=[(x_test, y_test3)],
                early_stopping_rounds=20)
score3 = model.score(x_test, y_test3)
print("score3 : %.4f" %(score3 * 100.0))
# print(model.feature_importances_)
y_pred_3 = model.predict(x_test)
mae3 = mean_absolute_error(y_test3, y_pred_3)
print('mae3 : %.4f' %(mae3))
y_pred3 = model.predict(x_pred)

model.fit(x_train, y_train4, verbose=False, eval_metric=['logloss'],
                eval_set=[(x_test, y_test4)],
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
sub.to_csv('./data/dacon/comp1/KB_lgbm_02.csv', index=False)