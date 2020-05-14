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

# 200514 0900~ 과제풀이 (r2_test_1)

'''

밥도 적당히 먹으면 좋은데 지나치게 먹으면 토한다. R2도 마찬가지
r2는 회계지표에서 acc와 유사하다 - 정확하게 같은 기능은 아니지만 대강 판단하는 것
자, acc 0 일때 r2 0 // (그래프 -x축 epochs 로 봤을 때)
쭈우욱 올라가다가 이후 떨어지는 경우가 생긴다. (과적합 구간 - 토하기 시작하는 시점)
loss는 첫 시점은 높은 값에서 내려오다 마찬가지로 요동친다.
그렇다면 적당한 기준은 어디일까? > 사람이 정한다.
기준을 정했을 때 그 기준이 맞다고 판단하는 법 > 1. 경험치를 쌓아야한다.
2. 쭈우욱 올라가다가 떨어지는 시점에 'if문 쓰든 뚝 떨어지는 시점에서 멈춰라' 라고 코딩을 한다.
[얼리 스타핑] > 코딩하게 되지 않을까? 구현되어 있다.
- 하지만 이것도 문제가 있다. 당한다음에 작동하는 것.? 이건 아직 이해안됨

'''