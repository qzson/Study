# 200623_26

# LGBM
# XGB 보다 속도가 훨씬 빠르다
# 다만, 1만행 이하 짜리 작은 것은 속도가 더 걸린다
# pip install LightGBM

# cancer / model save


import numpy as np
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score, recall_score, f1_score, precision_score, roc_auc_score
from sklearn.metrics import confusion_matrix


### 데이터 ###
x, y = load_breast_cancer(return_X_y=True)
print(x.shape)      # (569, 30)
print(y.shape)      # (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                 shuffle = True, random_state = 66)


### 기본 모델 ###
model = LGBMClassifier(n_estimators=300, learning_rate=0.1, n_jobs=-1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('acc :', score)

#== Default acc : 0.9736842105263158 ==#


### feature engineering ###
thresholds = np.sort(model.feature_importances_)
print(thresholds)
'''
[ 25  26  33  46  47  48  49  51  59  62  63  64  70  71  83  85  85  94
  95  98 110 112 112 124 128 138 146 168 175 302]
'''


for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    selection_model = LGBMClassifier(n_estimators=300, learning_rate=0.1, n_jobs=-1)
    selection_model.fit(select_x_train, y_train, verbose=False, eval_metric='logloss',
                eval_set=[(select_x_train, y_train), (select_x_test, y_test)],
                early_stopping_rounds=20)
    
    y_pred = selection_model.predict(select_x_test)

    acc = accuracy_score(y_test, y_pred)

    # get_clf_eval(y_test, y_pred)

    print('Thresh=%.3f, n=%d, acc: %.2f%%' %(thresh, select_x_train.shape[1], acc*100.0))
    # model.save_model('./model/xgb_save/cancer_n=%d_acc=%.3f.model' %(select_x_train.shape[1], acc))

def get_clf_eval(y_test, y_pred):
    confusion = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    print('오차행렬:\n', confusion)
    print('\n정확도: {:.4f}'.format(accuracy))
    print('정밀도: {:.4f}'.format(precision))
    print('재현율: {:.4f}'.format(recall))
    print('F1: {:.4f}'.format(F1))
    print('AUC: {:.4f}'.format(AUC))

get_clf_eval(y_test, y_pred)

# Thresh=25.000, n=30, acc: 96.49%
# Thresh=26.000, n=29, acc: 96.49%
# Thresh=33.000, n=28, acc: 95.61%
# Thresh=46.000, n=27, acc: 97.37%
# Thresh=47.000, n=26, acc: 96.49%
# Thresh=48.000, n=25, acc: 97.37%
# Thresh=49.000, n=24, acc: 96.49%
# Thresh=51.000, n=23, acc: 96.49%
# Thresh=59.000, n=22, acc: 97.37%
# Thresh=62.000, n=21, acc: 96.49%
# Thresh=63.000, n=20, acc: 97.37%
# Thresh=64.000, n=19, acc: 97.37%
# Thresh=70.000, n=18, acc: 97.37%
# Thresh=71.000, n=17, acc: 95.61%
# Thresh=83.000, n=16, acc: 95.61%
# Thresh=85.000, n=15, acc: 95.61%
# Thresh=85.000, n=15, acc: 95.61%
# Thresh=94.000, n=13, acc: 96.49%
# Thresh=95.000, n=12, acc: 96.49%
# Thresh=98.000, n=11, acc: 96.49%
# Thresh=110.000, n=10, acc: 94.74%
# Thresh=112.000, n=9, acc: 94.74%
# Thresh=112.000, n=9, acc: 94.74%
# Thresh=124.000, n=7, acc: 94.74%
# Thresh=128.000, n=6, acc: 94.74%
# Thresh=138.000, n=5, acc: 92.98%
# Thresh=146.000, n=4, acc: 92.98%
# Thresh=168.000, n=3, acc: 92.11%
# Thresh=175.000, n=2, acc: 91.23%
# Thresh=302.000, n=1, acc: 71.93%