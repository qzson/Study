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
# r2_y_predict = r2_score(y_test, y_predict) = 아래와 같음
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# 결론 : R2는 RMSE 와 같은 보조지표와 같이 쓴다.

'''
R2 or R^2 : R Squared, 결정계수
: (q1 - q2)/q1
q1 = 전체 데이터들의 편차의 제곱
q2 = 전체 데이터들의 잔차의 제곱
1에 가까울 수록 정확도가 높다
다른 지표들과 함께 사용된다.
https://jihongl.github.io/2017/09/16/Rsquared/
r2의 이해 링크
'''