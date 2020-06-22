# 200622_25
# time 함수


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


### 모델 ###
model = XGBRegressor()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('R2 :', score)

#== Default R2 : 0.9221188544655419 ==#


### feature engineering
thresholds = np.sort(model.feature_importances_)                        # 정렬될 것
print(thresholds)                                                       # 오름차순으로 정렬이 되었다

import time
start = time.time()

for thresh in thresholds:                                               # 컬럼수 만큼(13번) 돈다! 빙글 빙글
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                                # median 체크 할 것

    select_x_train = selection.transform(x_train)

    selection_model = XGBRegressor(n_estimators=1000)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)
    # print('R2 :', score)

    print('Thresh=%.3f, n=%d, R2: %.2f%%' %(thresh, select_x_train.shape[1], score*100.0))

# end = time.time() - start
# print('그냥 걸린 시간 :', end)
# >>> 위에 구성 시, 수정 할 필요 없음


start2 = time.time()

for thresh in thresholds:                                               # 컬럼수 만큼(13번) 돈다! 빙글 빙글
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                                # median 체크 할 것

    select_x_train = selection.transform(x_train)

    selection_model = XGBRegressor(n_jobs=6, n_estimators=1000)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)
    # print('R2 :', score)

    print('Thresh=%.3f, n=%d, R2: %.2f%%' %(thresh, select_x_train.shape[1], score*100.0))

end = start2 - start
print('그냥 걸린 시간 :', end)
end2 = time.time() - start2
print('jobs 걸린 시간 :', end2)
# >>> 아래에 전부 구성 시, 이렇게 기재해야 정확한 시간 비교 가능
# >>> 현재 코드 SFM에서는 n_jobs=-1이 먹히지 않는 것으로 확인
# n_jobs=6 코어 돌리는 것이 더 빠름 (현재 컴퓨터 코어수에 맞추어 전체 돌리기) - n_jobs 디폴트는 1이다 ?