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

# model.add(Dense(5, input_dim = 3))
model.add(Dense(5, input_shape =(3, )))

# 쉐이프, 디멘션
# input_dim =3 은 input_shape = (3,) 으로 바꿀 수 있다.
# 100행 3열 / 행 명시 x 열만 3하는 건데 엄밀히 틀린 것.
# 현재 상태에서는 그냥 그렇다고 알아두는 것 (앞으로 바뀐다.)

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