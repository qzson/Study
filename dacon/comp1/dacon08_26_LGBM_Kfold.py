# 200624

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


x_npy = np.load('./data/dacon/comp1/x_train.npy')
y_npy = np.load('./data/dacon/comp1/y_train.npy')
x_pred = np.load('./data/dacon/comp1/x_pred.npy')
print(x_npy.shape, y_npy.shape, x_pred.shape)   # (10000, 176) (10000, 4) (10000, 176)

x_train, x_test, y_train, y_test = train_test_split(x_npy, y_npy, test_size = 0.2, random_state = 66, shuffle = True)
print(x_train.shape)        # (8000, 176)
print(x_test.shape)         # (2000, 176)
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

model = LGBMRegressor(n_estimators=550,
                      num_leaves=100,
                      max_depth=20,
                      min_child_samples=30,
                      learning_rate=0.04,
                      colsample_bytree=0.5,
                      reg_alpha=1,
                      reg_lambda = 1.1,
                      n_jobs=6)

kf = KFold(n_splits=10, shuffle=True, random_state=66)

print('\n===== y : 1 ====== : \n')

for t_idx, v_idx in kf.split(x_train, y_train1):
    # print(t_idx.shape, v_idx.shape)
    xk_train, xk_test = x_train[t_idx], x_train[v_idx]
    yk_train, yk_test = y_train1[t_idx], y_train1[v_idx]
    model.fit(xk_train, yk_train, verbose=False, eval_metric=['logloss'],
                eval_set=[(xk_test, yk_test)],
                early_stopping_rounds=20)

    score1 = model.score(x_test, y_test1)
    print("score1 : %.4f" %(score1 * 100.0))
    # print(model.feature_importances_)
    y_pred_1 = model.predict(x_test)
    mae1 = mean_absolute_error(y_test1, y_pred_1)
    print('mae1 : %.4f' %(mae1))
    y_pred1 = model.predict(x_pred)


# print('\n=====기존모델====== : \n')
# model.fit(x_train, y_train1, verbose=False, eval_metric=['logloss'],
#                 eval_set=[(x_test, y_test1)],
#                 early_stopping_rounds=20)

# score1 = model.score(x_test, y_test1)
# print("score1 : %.4f" %(score1 * 100.0))
# # print(model.feature_importances_)
# y_pred_1 = model.predict(x_test)
# mae1 = mean_absolute_error(y_test1, y_pred_1)
# print('mae1 : %.4f' %(mae1))
# y_pred1 = model.predict(x_pred)

print('\n===== y : 2 ====== : \n')

for t_idx, v_idx in kf.split(x_train, y_train2):
    # print(t_idx.shape, v_idx.shape)
    xk_train, xk_test = x_train[t_idx], x_train[v_idx]
    yk_train, yk_test = y_train2[t_idx], y_train2[v_idx]
    model.fit(xk_train, yk_train, verbose=False, eval_metric=['logloss'],
                eval_set=[(xk_test, yk_test)],
                early_stopping_rounds=20)

    score2 = model.score(x_test, y_test2)
    print("score1 : %.4f" %(score2 * 100.0))
    # print(model.feature_importances_)
    y_pred_2 = model.predict(x_test)
    mae2 = mean_absolute_error(y_test2, y_pred_2)
    print('mae1 : %.4f' %(mae2))
    y_pred2 = model.predict(x_pred)

# model.fit(x_train, y_train2, verbose=False, eval_metric=['logloss'],
#                 eval_set=[(x_test, y_test2)],
#                 early_stopping_rounds=20)
                
# score2 = model.score(x_test, y_test2)
# print("score2 : %.4f" %(score2 * 100.0))
# # print(model.feature_importances_)
# y_pred_2 = model.predict(x_test)
# mae2 = mean_absolute_error(y_test2, y_pred_2)
# print('mae2 : %.4f' %(mae2))
# y_pred2 = model.predict(x_pred)

print('\n===== y : 3 ====== : \n')

for t_idx, v_idx in kf.split(x_train, y_train3):
    # print(t_idx.shape, v_idx.shape)
    xk_train, xk_test = x_train[t_idx], x_train[v_idx]
    yk_train, yk_test = y_train3[t_idx], y_train3[v_idx]
    model.fit(xk_train, yk_train, verbose=False, eval_metric=['logloss'],
                eval_set=[(xk_test, yk_test)],
                early_stopping_rounds=20)

    score3 = model.score(x_test, y_test3)
    print("score1 : %.4f" %(score3 * 100.0))
    # print(model.feature_importances_)
    y_pred_3 = model.predict(x_test)
    mae3 = mean_absolute_error(y_test3, y_pred_3)
    print('mae1 : %.4f' %(mae3))
    y_pred3 = model.predict(x_pred)

# model.fit(x_train, y_train3, verbose=False, eval_metric=['logloss'],
#                 eval_set=[(x_test, y_test3)],
#                 early_stopping_rounds=20)
# score3 = model.score(x_test, y_test3)
# print("score3 : %.4f" %(score3 * 100.0))
# # print(model.feature_importances_)
# y_pred_3 = model.predict(x_test)
# mae3 = mean_absolute_error(y_test3, y_pred_3)
# print('mae3 : %.4f' %(mae3))
# y_pred3 = model.predict(x_pred)

print('\n===== y : 4 ====== : \n')

for t_idx, v_idx in kf.split(x_train, y_train4):
    # print(t_idx.shape, v_idx.shape)
    xk_train, xk_test = x_train[t_idx], x_train[v_idx]
    yk_train, yk_test = y_train4[t_idx], y_train4[v_idx]
    model.fit(xk_train, yk_train, verbose=False, eval_metric=['logloss'],
                eval_set=[(xk_test, yk_test)],
                early_stopping_rounds=20)

    score4 = model.score(x_test, y_test4)
    print("score1 : %.4f" %(score4 * 100.0))
    # print(model.feature_importances_)
    y_pred_4 = model.predict(x_test)
    mae4 = mean_absolute_error(y_test4, y_pred_4)
    print('mae1 : %.4f' %(mae4))
    y_pred4 = model.predict(x_pred)

# model.fit(x_train, y_train4, verbose=False, eval_metric=['logloss'],
#                 eval_set=[(x_test, y_test4)],
#                 early_stopping_rounds=20)
# score4 = model.score(x_test, y_test4)
# print("score4 : %.4f" %(score4 * 100.0))
# # print(model.feature_importances_)
# y_pred_4 = model.predict(x_test)
# mae4 = mean_absolute_error(y_test4, y_pred_4)
# print('mae4 : %.4f' %(mae4))
# y_pred4 = model.predict(x_pred)


sub = pd.DataFrame({
    'id': np.array(range(10000, 20000)),
    'hhb': y_pred1,
    'hbo2': y_pred2,
    'ca': y_pred3,
    'na': y_pred4
})
print(sub)

''' 4. 저장 '''
sub.to_csv('./data/dacon/comp1/KB_lgbm_kfold_06.csv', index=False)