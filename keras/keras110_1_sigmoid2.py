# 20-07-02_27
# sigmoid 급조 파일


### 1. 데이터
import numpy as np

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,0,1,0,1,0,1,0,1,0])


### 2. 모델
from keras.models import Sequential, Model
from keras.layers import Dense, Input

model = Sequential()
model.add(Dense(128, input_shape=(1,)))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
# sigmoid를 위에 layer에 사용하지 않는 이유는?
# > 0과 1사이로 수렴시키는 것이 sigmoid 인데, 각 레이어 마다 0.5를 곱하면 0에 수렴하는 문제가 생긴다?
# relu는 식 자체가 무조건 0으로 수렴하는 것을 막아준다.


### 3. 실행, 훈련
model.compile(loss = ['binary_crossentropy'], optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)


### 4. 평가, 예측
loss = model.evaluate(x_train, y_train)
print('loss :', loss)

x1_pred = np.array([11, 12, 13, 14])

y_pred = model.predict(x1_pred)
print(y_pred)