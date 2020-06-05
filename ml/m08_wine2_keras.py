# 20-06-05

# 분류 모델 딥러닝 구성
# acc 70% 이상

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from keras.utils import np_utils


''' 1. 데이터 '''
winequality = pd.read_csv('./data/csv/winequality-white.csv',
                                index_col=None, header=0, sep = ';')
print(winequality)       
print(winequality.shape)            # (4898, 12)

# pandas -> numpy
winequality_npy = winequality.values
# print(type(winequality_npy))        # numpy

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
print(x_train.shape)                # 3918, 11
print(x_test.shape)                 # 980, 11

# y 원핫인코딩
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)                # (3918, 10)
print(y_test.shape)                 # (980, 10)


''' 2. 모델 '''
model = Sequential()

model.add(Dense(100, activation='relu', input_shape=(11,)))
model.add(Dense(350, activation = 'relu'))
model.add(Dense(500, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(350, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()


''' 3. 실행, 훈련 '''
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1,
          validation_split=0.25,
        #   shuffle=True,
          )


''' 4. 평가, 예측'''
loss, acc = model.evaluate(x_test, y_test, batch_size=64)
print('loss :', loss)
print('acc :', acc)

# loss : 3.1188064906061914
# acc : 0.6275510191917419

y_pred = model.predict(x_test)
# print(x_test, '의 예측 결과 \n', y_pred)


''' wine1 & wine2 결과 해석 '''
# acc가 잘나와봐야 70% 정도일 것이다. 이유는 ? 데이터에 사실 문제가 있다.
# iris나 다른 예제들은 공정한 분류가 가능하다. 하지만, 와인은 quality가 한 곳으로 몰려있다.
# 만약, 5와 6이 80%가 넘긴다. 그렇다면, 그냥 평타 80% 나오는거야 ~