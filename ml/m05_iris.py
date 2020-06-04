# 20-06-04
# iris data로 다중 분류 머신러닝 구성 및 모델 간 비교
# SCORE, ACC_SCORE 비교
# acc

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from keras.utils import np_utils

from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


''' 1. 데이터 '''
x, y = load_iris(return_X_y=True)

# 스플릿
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, shuffle=True, train_size=0.8)

# x 정규화
scale = StandardScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)


''' 2. 모델 '''
# model = LinearSVC()                            # score :  0.933 // acc =  0.933
# model = SVC()                                  # score :  0.933 // acc =  0.933
# model = KNeighborsRegressor(n_neighbors=1)     # score :  0.798 // acc =  0.866 // r2 : 0.798
# model = KNeighborsClassifier(n_neighbors=1)    # score :  0.866 // acc =  0.866
# model = RandomForestRegressor()                # error // score, r2 : 0.961
model = RandomForestClassifier()               # score :  0.966 // acc =  0.966


''' 3. 실행, 훈련 '''
model.fit(x_train, y_train)

score = model.score(x_test, y_test) # 케라스 딥러닝의 evaluate와 같다. 그러므로 테스트를 넣어야한다.
print("score : ", score)


''' 4. 평가, 예측 '''
y_predict = model.predict(x_test)

# acc
acc = accuracy_score(y_test, y_predict)
print('acc = ', acc)

# r2 (분류 모델에서 r2 보조지표를 쓰는 오류를 범하지 말자)
# r2 = r2_score(y_test, y_predict)
# print("R2 : ", r2)

print(y_test, '의 예측 결과', '\n', y_predict)
