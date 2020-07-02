## 1. 데이터
import numpy as np
x = np.array([1,2,3,4])
y = np.array([1,2,3,4])


## 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(11))
model.add(Dense(1))

from keras.optimizers import Adam, RMSprop, SGD, adadelta, adagrad, nadam
# 경사하강법의.. 베이스를 두고 있는.. adam.. 성능이 꽤 좋다.

ad = Adam(lr=0.001)         # 0.06930536776781082
rms = RMSprop(lr=0.001)     # 0.10394549369812012
sd = SGD(lr=0.001)          # 0.020946599543094635
adad = adadelta(lr=0.001)   # 10.468718528747559
adag = adagrad(lr=0.001)    # 3.564135789871216
na = nadam(lr=0.001)        # 0.01681675761938095

# > 그리드서치에 경우의 수도 또 늘어날 것 (optimizer, learning_rate)
# 이놈들의 기반은 약간씩은 차이있지만, 경사하강법에 기초를 두고 있다.

## 3. 훈련, 평가
model.compile(loss='mse', optimizer=na, metrics=['mse'])

model.fit(x, y, epochs=100, verbose=2)

loss, mse = model.evaluate(x, y)
print('{:.5f}'.format(loss))
# print(loss)

## 4. 예측
pred1 = model.predict([3.5])
print(pred1)