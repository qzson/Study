# 200622_25
# boston / SFM

'''
m28_eval 1~3 만들고
1. eval 에 'loss' 와 다른 지표 1개 더 추가.
2. earlyStopping 적용
3. plot 으로 그릴 것.

SelectFromModel 에 적용 (그래프 없이)
1) 회귀         m29_eval1
2) 이진 분류    m29_eval2
3) 다중 분류    m29_eval3

4. 결과는 주석으로 소스 하단에 표시.
5. m27 ~ 29 까지 완벽 이해할 것!
'''


from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, r2_score


### 데이터 ###
x, y = load_boston(return_X_y=True)
print(x.shape)      # (506, 13)
print(y.shape)      # (506, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                 shuffle = True, random_state = 66)


### 기본 모델 ###
model = XGBRegressor(n_estimators=300, learning_rate=0.1, n_jobs=-1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('R2 :', score)

#== Default R2 : 0.9313126937746082 ==#


### feature engineering ###
thresholds = np.sort(model.feature_importances_)
print(thresholds)
'''
[0.00097099 0.00200123 0.01014188 0.01031922 0.01329956 0.01377407
 0.01690491 0.02635994 0.03570918 0.04228422 0.04502721 0.2457639
 0.53744376]
'''

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    selection_model = XGBRegressor(n_estimators=300, learning_rate=0.1, n_jobs=-1)
    selection_model.fit(select_x_train, y_train, verbose=False, eval_metric=['logloss','rmse'],
                eval_set=[(select_x_train, y_train), (select_x_test, y_test)],
                early_stopping_rounds=20)
    
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)

    print('Thresh=%.3f, n=%d, R2: %.2f%%' %(thresh, select_x_train.shape[1], score*100.0))

# Thresh=0.001, n=13, R2: 93.29%
# Thresh=0.002, n=12, R2: 93.25%
# Thresh=0.010, n=11, R2: 93.27%
# Thresh=0.010, n=10, R2: 93.40%
# Thresh=0.013, n=9, R2: 93.20%
# Thresh=0.014, n=8, R2: 93.49%
# Thresh=0.017, n=7, R2: 93.52%
# Thresh=0.026, n=6, R2: 94.02%
# Thresh=0.036, n=5, R2: 92.82%
# Thresh=0.042, n=4, R2: 92.00%
# Thresh=0.045, n=3, R2: 89.06%
# Thresh=0.246, n=2, R2: 81.25%
# Thresh=0.537, n=1, R2: 68.26%
