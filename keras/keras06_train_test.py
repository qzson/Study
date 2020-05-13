# 1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16, 17, 18])
'''
이것만 보더라도 y_pred 보고싶다 // 이걸 전이학습
'''

# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 1))
model.add(Dense(25))
model.add(Dense(50))
model.add(Dense(250))
model.add(Dense(500))
model.add(Dense(250))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(1))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=200, batch_size=1)

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test)
print("loss :", loss)
print("mse :", mse)

y_pred = model.predict(x_pred)
print("y_predic :", y_pred)

'''

레이어의 깊이 /  노드의 갯수 / 에포 / 뱃치 사이즈 /
를 통해 로스를 낮춰 근사값을 만들어라.

오후 시간 ~
트레인, 테스트 데이터 분류는 우리가 한다.

x ([학생수능점수, 온도, 날씨, 하이])
10달치 데이타 를 트레인을 70%한다면 7달을 트레인을 한다. 테스트는 3달
트레인은 모델.핏 / 테스트는 모델.프레딕트로 평가를 한다.
트레인과 테스트 데이터를 나누는 이유는 7대 3으로 먼저 트레인 후 나머지로 평가를 한 후
나중에 쌩뚱맞은 데이터를 받아도 결과치를 최대한 끌어올릴 가능성이 높아지기 때문에 이렇게 한다.?
평가데이터는 모델에 반영되지 않는다. 평가는 보기 위한 것.
하지만 트레인 데이터는 매트릭스 보면 훈련에 반영이 된다.

'''