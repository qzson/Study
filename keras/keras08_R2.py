# 1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16, 17, 18])

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

# y_pred = model.predict(x_pred)
# print("y_predic :", y_pred)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error # mse 를 sklearn.metrics 에서 땡겨오겠다.
def RMSE(y_test, y_predict): # RMSE로 정의하겠다. 원래 y값 테스트와 y프레딕트를 땡겨올거고
    return np.sqrt(mean_squared_error(y_test, y_predict)) # np.스쿼트(루트)를 리턴해줄거다
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
# r2_y_predict = r2_score(y_test, y_predict) =
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# 결론 : R2는 RMSE 와 같은 보조지표와 같이 쓴다.

# 과제 : R2를 음수가 아닌 0.5 이하로 줄이기
# 레이어는 인풋과 아웃풋을 포함 5개 이상(히든이 3개 이상), 히든레이어 노드는 레이어당 각각 최소 5개 이상
# batch_size = 1
# epochs = 100 이상
# 데이터 조작 하지말아라

'''

이번에는 강제로 나쁜 모델을 만들어야한다
이 조건을 갖춘 상황에서 R2 스코어가 0.5 이하로 나와야 한다.

'''