# 20-06-10_22 수요일 // 0900~

# 파이프라인의 궁극적인 목적
# 크로스발리데이션
# 파이프라인을 통해서 검증하는 부분까지 같이 잡아준다면 깔끔하게 노드가 돌아갈 때마다 전처리가 된다
# 이전에 파이프라인에 param에 오차가 있었다 그래서 오늘 정리를 할 것이다
# + 오늘 시간에는 케라스에 연결을 할 것이다

import warnings
warnings.filterwarnings("ignore")

import pandas as pds
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
    {'svc__C':[1, 10, 100, 1000], 'svc__kernel':['linear']},    # 4 가지
    {'svc__C':[1, 10, 100], 'svc__kernel':['rbf'], 'svc__gamma':[0.001, 0.0001]}, # 8 가지
    {'svc__C':[1, 100, 1000], 'svc__kernel':['sigmoid'], 'svc__gamma':[0.001, 0.0001]}  # 8 가지
]
# 총 경우의 수 20 가지
# 'svm__C' 자리에 [1, 10, 100, 1000] 을 동일하게 3라인 가져가고
# 'svm__C' 자리에 [1, 100, 1000] 으로 구성 하면 경우의 수가 바뀐다


''' 2. 모델 '''
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# pipe = Pipeline([('scaler', MinMaxScaler()), ('svm', SVC())])
pipe = make_pipeline(MinMaxScaler(), SVC())
#   >> 위 파라미터 요소들을 'svm__C' 를 'svc__C' 형태로 바꿔주면 가능 (즉, 모델)


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