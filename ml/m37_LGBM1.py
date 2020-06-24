# 200623_26

# LGBM
# XGB 보다 속도가 훨씬 빠르다
# 다만, 1만행 이하 짜리 작은 것은 속도가 더 걸린다
# pip install LightGBM

# boston / model save


import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score, recall_score, f1_score, precision_score, roc_auc_score
from sklearn.metrics import confusion_matrix


### 데이터 ###
x, y = load_boston(return_X_y=True)
print(x.shape)      # (506, 13)
print(y.shape)      # (506, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                 shuffle = True, random_state = 66)


### 기본 모델 ###
model = LGBMRegressor(n_estimators=300, learning_rate=0.1, n_jobs=-1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('R2 :', score)

#== Default R2 : 0.9205665502067463 ==#


### feature engineering ###
thresholds = np.sort(model.feature_importances_)
print(thresholds)
'''
[ 16  25  72 116 149 180 273 470 518 601 616 706 749]
'''

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    selection_model = LGBMRegressor(n_estimators=300, learning_rate=0.1, n_jobs=-1)
    selection_model.fit(select_x_train, y_train, verbose=False, eval_metric=['logloss','rmse'],
                eval_set=[(select_x_train, y_train), (select_x_test, y_test)],
                early_stopping_rounds=20)
    
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)

    print('Thresh=%.3f, n=%d, R2: %.2f%%' %(thresh, select_x_train.shape[1], score*100.0))
    # model.save_model('./model/xgb_save/boston_n=%d_r2=%.3f.model' %(select_x_train.shape[1], score))

# Thresh=16.000, n=13, R2: 92.80%
# Thresh=25.000, n=12, R2: 92.87%
# Thresh=72.000, n=11, R2: 92.70%
# Thresh=116.000, n=10, R2: 92.74%
# Thresh=149.000, n=9, R2: 92.54%
# Thresh=180.000, n=8, R2: 91.95%
# Thresh=273.000, n=7, R2: 90.93%
# Thresh=470.000, n=6, R2: 89.75%
# Thresh=518.000, n=5, R2: 88.56%
# Thresh=601.000, n=4, R2: 88.31%
# Thresh=616.000, n=3, R2: 87.48%
# Thresh=706.000, n=2, R2: 83.49%
# Thresh=749.000, n=1, R2: 75.24%
