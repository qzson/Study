# 과제
# <R2를 음수가 아닌 0.5 이하로 줄이기>
# 1. 레이어는 인풋과 아웃풋을 포함 5개 이상(히든이 3개 이상), 히든레이어 노드는 레이어당 각각 최소 5개 이상
# 2. batch_size = 1
# 3. epochs = 100 이상
# 4. 데이터 조작 금지

'''
이번에는 강제로 나쁜 모델을 만들어야한다. 위 조건을 갖춘 상황에서 R2 스코어가 0.5 이하로 나와야 한다.
'''

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

model.add(Dense(15, input_dim = 1))
model.add(Dense(5))
model.add(Dense(1000))
model.add(Dense(5))
model.add(Dense(1000))
model.add(Dense(5))
model.add(Dense(1000))
model.add(Dense(5))
model.add(Dense(1000))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(1005))
model.add(Dense(5))
model.add(Dense(1000))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(100))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(1))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=252, batch_size=1)

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test)
print("loss :", loss)
print("mse :", mse)


y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)