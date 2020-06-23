# 200623_26

# LGBM
# XGB 보다 속도가 훨씬 빠르다
# 다만, 1만행 이하 짜리 작은 것은 속도가 더 걸린다
# pip install LightGBM

# iris / model save


import numpy as np
from sklearn.datasets import load_boston, load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score, recall_score, f1_score, precision_score, roc_auc_score
from sklearn.metrics import confusion_matrix


### 데이터 ###
x, y = load_iris(return_X_y=True)
print(x.shape)      # (150, 4)
print(y.shape)      # (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                 shuffle = True, random_state = 66)


### 기본 모델 ###
model = LGBMClassifier(objective='multi:softmax', n_estimators=300, learning_rate=0.1, n_jobs=-1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('acc :', score)

#== Default acc : 0.9333333333333333 ==#


### feature engineering ###
thresholds = np.sort(model.feature_importances_)
print(thresholds)
'''
[574 594 634 915]
'''

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    selection_model = LGBMClassifier(objective='multi:softmax', n_estimators=300, learning_rate=0.1, n_jobs=-1)
    selection_model.fit(select_x_train, y_train, verbose=False, eval_metric=['multi_logloss', 'multi_error'],
                eval_set=[(select_x_train, y_train), (select_x_test, y_test)],
                early_stopping_rounds=20)
    
    y_pred = selection_model.predict(select_x_test)

    acc = accuracy_score(y_test, y_pred)

    print('Thresh=%.3f, n=%d, acc: %.2f%%' %(thresh, select_x_train.shape[1], acc*100.0))
    # model.save_model('./model/xgb_save/iris_n=%d_acc=%.3f.model' %(select_x_train.shape[1], acc))

# Thresh=574.000, n=4, acc: 96.67%
# Thresh=594.000, n=3, acc: 93.33%
# Thresh=634.000, n=2, acc: 73.33%
# Thresh=915.000, n=1, acc: 50.00%