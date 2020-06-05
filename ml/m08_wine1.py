# 20-06-05

# 분류 모델 머신러닝 구성
# acc 70% 이상

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from keras.utils import np_utils

from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


''' 1. 데이터 '''
winequality = pd.read_csv('./data/csv/winequality-white.csv',
                                index_col=None, header=0, sep = ';')
print(winequality)       
print(winequality.shape)            # (4898, 12)

# pandas -> numpy
winequality_npy = winequality.values
print(winequality_npy.shape)
print(type(winequality_npy))        # numpy

x = winequality_npy[:, 0:11]
y = winequality_npy[:, 11]

# print(x)
# print(y)
print(x.shape)                      # (4898, 11)
print(y.shape)                      # (4898, )

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
model = KNeighborsClassifier(n_neighbors=1)    # score :  0.866 // acc =  0.866
# model = RandomForestRegressor()                # error // score, r2 : 0.961
# model = RandomForestClassifier()               # score :  0.702


''' 3. 실행, 훈련 '''
model.fit(x_train, y_train)

score = model.score(x_test, y_test) # 케라스 딥러닝의 evaluate와 같다. 그러므로 테스트를 넣어야한다.
print("score : ", score)


''' 4. 평가, 예측 '''
y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)
print('acc = ', acc)

# print(y_test, '의 예측 결과', '\n', y_predict)


''' wine1 & wine2 결과 해석 '''
# acc가 잘나와봐야 70% 정도일 것이다. 이유는 ? 데이터에 사실 문제가 있다.
# iris나 다른 예제들은 공정한 분류가 가능하다. 하지만, wine data는 quality가 한 곳으로 몰려있다.
# 만약, 5와 6이 80%가 넘긴다. 그렇다면, 그냥 평타 80% 나오는거야 ~