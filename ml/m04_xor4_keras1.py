# 200604

# 머신러닝 모델을 신경망 레이어 단 1개 케라스로 구현해보자 (딥러닝 없이)
# 딥러닝에서 행렬 연산 때문에 numpy를 사용한다.
# 머신러닝에서는 가중치 연산이 아니여서 그냥 리스트도 가능하다.
# >> 글씨도 그냥 수치로 바꾼다. 라벨 인코더 ?
# 라벨 인코더에 데이터셋을 넣으면 사람은 1 고양이는 2 이런 식으로 알아서 해준다.
# 머신 러닝은 그냥 되는데 케라스에서는 레이어마다 가중치 연산을 하기 때문에 numpy로 바꿔줘야 한다.


import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x_data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_data = np.array([0, 1, 1, 0])

print(x_data.shape)
print(y_data.shape)

# 2. 모델
model = Sequential()
# model.add(Dense(1, input_dim=2))
# model.add(Dense(1, activation='sigmoid')) # 이렇게 들어가면, 신경망 구조가 2-1-1 이 된다. (딥러닝 구조) 따라서, 없앤다
model.add(Dense(1, activation='sigmoid', input_shape=(2,)))

model.summary()

# 3. 훈련
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
model.fit(x_data, y_data, epochs=100, batch_size=1)


# 4. 평가, 예측
loss, acc = model.evaluate(x_data, y_data, batch_size=1)
print("loss :", loss)
print("acc :", acc)

x_test = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_predict = model.predict(x_test)
print(y_predict)


### 이전 머신러닝 훈련과 예측 부분 ###

# # 3. 훈련
# model.fit(x_data, y_data)

# # 4. 평가, 예측
# x_test = [[0, 0], [1, 0], [0, 1], [1, 1]]
# y_predict = model.predict(x_test)
# acc = accuracy_score([0, 1, 1, 0], y_predict)

# print(x_test, '의 예측 결과', y_predict)
# print('acc = ', acc)