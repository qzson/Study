# 200619_24
# iris, xgb, selectfrommodel


from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston, load_breast_cancer, load_iris
from sklearn.metrics import accuracy_score, r2_score

x, y = load_iris(return_X_y=True)
print(x.shape)      # (150, 4)
print(y.shape)      # (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                 shuffle = True, random_state = 66)

model = XGBClassifier()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('acc :', score)

#== Default acc : 0.9 ==#

### feature engineering

thresholds = np.sort(model.feature_importances_)                        # 정렬될 것
print(thresholds)                                                       # 오름차순으로 정렬이 되었다
'''
[0.01759811 0.02607087 0.33706376 0.6192673 ]
'''

for thresh in thresholds:                                               # 컬럼수 만큼(13번) 돈다! 빙글 빙글
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                                # median 체크 할 것

    select_x_train = selection.transform(x_train)

    selection_model = XGBClassifier()
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = accuracy_score(y_test, y_pred)
    # print('acc :', score)

    print('Thresh=%.3f, n=%d, acc: %.2f%%' %(thresh, select_x_train.shape[1], score*100.0))

'''
Thresh=0.018, n=4, acc: 90.00%
Thresh=0.026, n=3, acc: 90.00%
Thresh=0.337, n=2, acc: 96.67%
Thresh=0.619, n=1, acc: 93.33%
'''

# 1 xgboost 파라미터 이해
# 2 매개변수 thresh, median 이해
# 3 m24에 그리드 서치를 엮어라

# 200619 금 주말과제
# 데이콘 적용해라. 71개 컬럼
# 월요일 까지 소스 제출, 본인들의 성적을 메일로 발송
# 메일 제목 : 말똥이 24등