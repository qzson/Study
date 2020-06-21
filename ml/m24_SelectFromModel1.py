# 200619_24
# boston, xgb, selectfrommodel


from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, r2_score

x, y = load_boston(return_X_y=True)
print(x.shape)      # (506, 13)
print(y.shape)      # (506, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                 shuffle = True, random_state = 66)

model = XGBRegressor()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('R2 :', score)

#== Default R2 : 0.9221188544655419 ==#

### feature engineering

thresholds = np.sort(model.feature_importances_)                        # 정렬될 것
print(thresholds)                                                       # 오름차순으로 정렬이 되었다
'''
[0.00134153 0.00363372 0.01203115 0.01220458 0.01447935 0.01479119
 0.0175432  0.03041655 0.04246345 0.0518254  0.06949984 0.30128643
 0.42848358]
'''

for thresh in thresholds:                                               # 컬럼수 만큼(13번) 돈다! 빙글 빙글
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                                # median 체크 할 것

    select_x_train = selection.transform(x_train)
    # print(select_x_train.shape)
    # print(select_x_train[0])
    # 칼럼을 줄이면서 뭔가를 하고 있다 (중요하지 않은 것은 제거하면서?)
    # 13을 돌아가면서 컬럼의 중요도가 가장 높은 것을 빼낸다 (윤영선 선생님의 생각)

    selection_model = XGBRegressor()
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)
    # print('R2 :', score)

    print('Thresh=%.3f, n=%d, R2: %.2f%%' %(thresh, select_x_train.shape[1], score*100.0))

'''
Thresh=0.001, n=13, R2: 92.21%
Thresh=0.004, n=12, R2: 92.16%
Thresh=0.012, n=11, R2: 92.03%
Thresh=0.012, n=10, R2: 92.19%
Thresh=0.014, n=9, R2: 93.08%
Thresh=0.015, n=8, R2: 92.37%
Thresh=0.018, n=7, R2: 91.48%
Thresh=0.030, n=6, R2: 92.71%
Thresh=0.042, n=5, R2: 91.74%
Thresh=0.052, n=4, R2: 92.11%
Thresh=0.069, n=3, R2: 92.52%
Thresh=0.301, n=2, R2: 69.41%
Thresh=0.428, n=1, R2: 44.98%
'''

# Thresh : feature_importance 값 // (중요도에 따라 쭉 나오는 것)
# n=     : 중요도 제일 낮은 것을 뺀것 후 나머지
# r2     : 93.08% , 9개로 했을때 좋았다


# 1 xgboost 파라미터 이해
# 2 매개변수 thresh, median 이해
# 3 m24에 그리드 서치를 엮어라

# 200619 금 주말과제
# 데이콘 적용해라. 71개 컬럼
# 월요일 까지 소스 제출, 본인들의 성적을 메일로 발송
# 메일 제목 : 말똥이 24등