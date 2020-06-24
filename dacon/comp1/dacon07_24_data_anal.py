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



train = pd.read_csv('./data/dacon/comp1/train.csv', index_col='id')
test = pd.read_csv('./data/dacon/comp1/test.csv', index_col='id')
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', index_col='id')

print('train.shape :', train.shape)             # (10000, 75) : x,y_train, test
print('test.shape :', test.shape)               # (10000, 71) : x_predict
print('sub.shape :', submission.shape)          # (10000,  4) : y_predict
print(train.head(10))

feature_names=list(test)
target_names =list(submission)
print(feature_names)
print(target_names)

x_train = train[feature_names]
y_train = train[target_names]
x_test = test[feature_names]

y_train1 = y_train['hhb']
y_train2 = y_train['hbo2']
y_train3 = y_train['ca']
y_train4 = y_train['na']