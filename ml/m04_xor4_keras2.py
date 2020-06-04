# 200604

# keras 레이어 늘려 acc 1만들기

from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

import numpy as np
from keras.models import Sequential
from keras.layers import Dense


# 1. 데이터
x_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_data = [0, 1, 1, 0]

x_data = np.array(x_data)
y_data = np.array(y_data)

print(x_data.shape)     # (4, 2)
print(y_data.shape)     # (4,  )

# (기존 머신러닝 여러가지 정의 방법)
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier(n_neighbors=1)
# lin = LinearSVC()
# sv = SVC()
# kn = KNeighborsClassifier()

# 2. 모델
model = Sequential()

model.add(Dense(10, activation='relu', input_dim=2))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()


# 3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=300)


# 4. 평가, 예측
loss, acc = model.evaluate(x_data, y_data)

x_test = [[0, 0], [1, 0], [0, 1], [1, 1]]
x_test = np.array(x_test)
y_predict = model.predict(x_test)
# y_predict = np.where(y_predict > 0.5, 1, 0)
    # 값이 0.5 보다 크면 1, 아니면 0
    # 이렇게 결과를 도출해내는 것

print(x_test, '의 예측 결과 \n', y_predict)
print('loss :', loss)
print('acc :', acc)