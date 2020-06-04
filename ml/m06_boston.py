# 20-06-04
# boston data로 회귀 머신러닝 구성 및 모델 간 비교
# SCORE 와 R2 score 비교 (회귀 모델)
# R2

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from keras.utils import np_utils

from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


''' 1. 데이터 '''
x, y = load_boston(return_X_y=True)

# 스플릿
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, shuffle=True, train_size=0.8)

# x 정규화
scale = StandardScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)


''' 2. 모델 '''
# model = LinearSVC()                            # error "ValueError: Unknown label type: 'continuous'"
# model = SVC()                                  # error
# model = KNeighborsRegressor(n_neighbors=1)     # score : 0.808 // R2 = 0.808
# model = KNeighborsClassifier(n_neighbors=1)    # error
model = RandomForestRegressor()                # score : 0.921 // R2 = 0.921
# model = RandomForestClassifier()               # error


''' 3. 실행, 훈련 '''
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("score : ", score)


''' 4. 평가, 예측 '''
y_predict = model.predict(x_test)

# # acc (회귀 모델에서 acc 보조지표를 쓰는 오류를 범하지 말자)
# acc = accuracy_score(y_test, y_predict)
# print('acc = ', acc)

# r2
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

print(y_test, '의 예측 결과', '\n', y_predict)
