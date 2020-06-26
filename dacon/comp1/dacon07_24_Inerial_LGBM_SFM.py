# 200624

# LGBM


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV, GridSearchCV, cross_val_predict
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

lgbm_model = LGBMRegressor()

model = MultiOutputRegressor(lgbm_model)
model.fit(x_train,y_train)
# print(model.estimators_)
# print(len(model.estimators_))
# print(model.estimators_[0].feature_importances_)

for i in range(len(model.estimators_)):
    threshold = np.sort(model.estimators_[i].feature_importances_)[18:]
    print(threshold)
    for thresh in threshold:
        selection = SelectFromModel(model.estimators_[i], threshold=thresh, prefit=True)

        param = {
            'n_estimators': [550],
            'num_leaves': [100],
            'max_depth': [20],
            'min_child_samples': [30],
            'learning_rate': [0.04],
            'colsample_bytree': [0.5],
            'reg_alpha': [1],
            'reg_lambda': [1.1]
        }
        gridcv = GridSearchCV(LGBMRegressor(n_jobs = 6), param, cv=5, n_jobs = 6)
        
        selection_model = MultiOutputRegressor(gridcv)
        
        select_x_train = selection.transform(x_train)
        selection_model.fit(select_x_train, y_train)
        print(selection_model.estimators_[i].best_params_)
        
        select_x_test = selection.transform(x_test)
        y_pred = selection_model.predict(select_x_test)

        mae = mean_absolute_error(y_test, y_pred)
        score = r2_score(y_test, y_pred)
        print('thresh=%.3f, n=%d, R2: %.2f%%, MAE: %.3f' %(thresh, select_x_train.shape[1], score*100.0, mae))

        select_x_pred = selection.transform(x_pred)
        y_predict = selection_model.predict(select_x_pred)

        a = np.arange(10000,20000)
        submission = pd.DataFrame(y_predict, a)
        submission.to_csv('./data/dacon/comp1/KB_LGBM_%i_%.5f.csv' %(i, mae), index = True, header=['hhb','hbo2','ca','na'], index_label='id')