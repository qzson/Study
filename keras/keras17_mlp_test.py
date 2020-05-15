# 200515, 1612 ~ /

# R2 값을 낮추는 안좋은 모델 만들기 <실습>

# 1. R2         : 0.5 이하
# 2. layer      : 5개 이상
# 3. node       : 10개 이상
# 4. batch_size : 8 이하
# 5. epochs     : 30 이상


# 1. 데이터
import numpy as np
x = np.transpose([range(1,101), range(311, 411), range(100)])
y = np.array(range(711,811))

# print(x.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = False,
    train_size = 0.8)

# print(x_train)
# print(x_test)


# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(10,input_dim=3))
model.add(Dense(6666))
model.add(Dense(10))
model.add(Dense(6666))
model.add(Dense(10))
model.add(Dense(6666))
model.add(Dense(10))
model.add(Dense(6666))
model.add(Dense(10))
model.add(Dense(6666))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=30, batch_size=8, verbose=2,
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
