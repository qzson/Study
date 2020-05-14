# 200514 16:00 ~
# ※ 어려웠음 이거 데이터 구성 split 한개만 사용, validation data 지우고

# 1. 데이터
import numpy as np
x = np.array(range(1,101))
y = np.array(range(101,201))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = False,
    train_size = 0.8)

print(x_train) # 1~80
# print(x_val) : validation_split을 사용하기 때문에 값을 지정해줄 수 없다.
print(x_test) # 81~100

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
model.add(Dense(5))
model.add(Dense(1))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
          #validation_data = (x_val, y_val)
          validation_split=0.25)
# validation_split 사용하여 validation 비율 설정 // 둘 중 뭘 사용하던 상관 없음.

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

# y = wx+b
# 181  = 1*81+100