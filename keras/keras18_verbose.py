# 200515, 1512 ~ / 

# 1. 데이터
import numpy as np
x = np.array([range(1,101), range(311, 411), range(100)])
y = np.array(range(711,811))

x = np.transpose(x)

print(x.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = False,
    train_size = 0.8)

print(x_train)
print(x_test)


# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 3))
model.add(Dense(25))
model.add(Dense(50))
model.add(Dense(250))
model.add(Dense(500))
model.add(Dense(250))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(5))
model.add(Dense(1))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
            verbose=2,
          validation_split=0.25)
# 0.01초 정도 딜레이를 한다. 훈련 과정을 보여주기 때문에, 그래서 verbose를 적용한다.
# verbose = 0 : 훈련 과정 노출 없음 (전부 보여주지 않는다. 하지만 돌아가는 중)
# verbose = 1 : Default 값. (다 보여주는 것)
# verbose = 2 : 프로세스 바를 생략 (조금 간소화)
# verbose = 3 : 로스 같은 값들도 생략 (제일 간소화)


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

# 다음 주에, 함수형과 분류 모델을 할 것이다.