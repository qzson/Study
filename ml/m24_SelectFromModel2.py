# 200619_24
# cancer, xgb, selectfrommodel


from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.metrics import accuracy_score, r2_score

x, y = load_breast_cancer(return_X_y=True)
print(x.shape)      # (569, 30)
print(y.shape)      # (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                 shuffle = True, random_state = 66)

model = XGBClassifier()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('acc :', score)

#== Default acc : 0.9736842105263158 ==#

### feature engineering

thresholds = np.sort(model.feature_importances_)                        # 정렬될 것
print(thresholds)                                                       # 오름차순으로 정렬이 되었다
'''
[0.         0.         0.00037145 0.00233393 0.00278498 0.00281184
 0.00326043 0.00340272 0.00369179 0.00430626 0.0050556  0.00513449
 0.0054994  0.0058475  0.00639412 0.00769184 0.00775311 0.00903706
 0.01171023 0.0136856  0.01420499 0.01813928 0.02285903 0.02365488
 0.03333857 0.06629944 0.09745205 0.11586285 0.22248562 0.28493083]
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
Thresh=0.000, n=30, acc: 97.37%
Thresh=0.000, n=30, acc: 97.37%
Thresh=0.000, n=28, acc: 97.37%
Thresh=0.002, n=27, acc: 97.37%
Thresh=0.003, n=26, acc: 97.37%
Thresh=0.003, n=25, acc: 97.37%
Thresh=0.003, n=24, acc: 97.37%
Thresh=0.003, n=23, acc: 97.37%
Thresh=0.004, n=22, acc: 96.49%
Thresh=0.004, n=21, acc: 96.49%
Thresh=0.005, n=20, acc: 97.37%
Thresh=0.005, n=19, acc: 97.37%
Thresh=0.005, n=18, acc: 96.49%
Thresh=0.006, n=17, acc: 96.49%
Thresh=0.006, n=16, acc: 96.49%
Thresh=0.008, n=15, acc: 97.37%
Thresh=0.008, n=14, acc: 97.37%
Thresh=0.009, n=13, acc: 98.25%
Thresh=0.012, n=12, acc: 98.25%
Thresh=0.014, n=11, acc: 98.25%
Thresh=0.014, n=10, acc: 98.25%
Thresh=0.018, n=9, acc: 97.37%
Thresh=0.023, n=8, acc: 97.37%
Thresh=0.024, n=7, acc: 98.25%
Thresh=0.033, n=6, acc: 97.37%
Thresh=0.066, n=5, acc: 95.61%
Thresh=0.097, n=4, acc: 96.49%
Thresh=0.116, n=3, acc: 94.74%
Thresh=0.222, n=2, acc: 91.23%
Thresh=0.285, n=1, acc: 88.60%
'''

# 1 xgboost 파라미터 이해
# 2 매개변수 thresh, median 이해
# 3 m24에 그리드 서치를 엮어라

# 200619 금 주말과제
# 데이콘 적용해라. 71개 컬럼
# 월요일 까지 소스 제출, 본인들의 성적을 메일로 발송
# 메일 제목 : 말똥이 24등