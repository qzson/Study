# 20-06-08_20
# 월요일 // 14:30 ~

# Pipeline
# m13 copy


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


''' 1. 데이터 '''
iris = load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 43, shuffle = True)

# 그리드 / 랜덤 서치에서 사용할 매개 변수
#   - 리스트 형태의 키:벨류 딕셔너리
parameters = [
    {'svm__C':[1, 10, 100, 1000], 'svm__kernel':['linear']},    # 4 가지
    {'svm__C':[1, 10, 100], 'svm__kernel':['rbf'], 'svm__gamma':[0.001, 0.0001]}, # 8 가지
    {'svm__C':[1, 100, 1000], 'svm__kernel':['sigmoid'], 'svm__gamma':[0.001, 0.0001]}  # 8 가지
]
# 총 경우의 수 20 가지
# 'svm__C' 자리에 [1, 10, 100, 1000] 을 동일하게 3라인 가져가고
# 'svm__C' 자리에 [1, 100, 1000] 으로 구성 하면 경우의 수가 바뀐다


''' 2. 모델 '''
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
pipe = Pipeline([('scaler', MinMaxScaler()), ('svm', SVC())])
# pipe = make_pipeline(MinMaxScaler(), SVC())
#   >> 위 파라미터 요소들을 'svm__C' 를 'svc__C' 형태로 바꿔주면 가능

model = RandomizedSearchCV(pipe, parameters, cv=5)


''' 3. 훈련 '''
model.fit(x_train, y_train)


''' 4. 평가, 예측 '''
acc = model.score(x_train, y_train)

print('최적의 매개변수 :', model.best_estimator_)
print('acc :', acc)

import sklearn as sk
print('sklearn :', sk.__version__)

# 최적의 매개변수 : Pipeline(memory=None,
#          steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))),
#                 ('svm',
#                  SVC(C=1, break_ties=False, cache_size=200, class_weight=None,
#                      coef0=0.0, decision_function_shape='ovr', degree=3,
#                      gamma='scale', kernel='linear', max_iter=-1,
#                      probability=False, random_state=None, shrinking=True,
#                      tol=0.001, verbose=False))],
#          verbose=False)
# acc : 0.9666666666666667
# sklearn : 0.22.1