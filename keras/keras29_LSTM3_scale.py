# 20.05.21 1142~

'''
실습1

튜닝 결과값 : y_predict : [[80.0025]]
'''

from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# import numpy as np
# x = np.array

# 1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = array([50,60,70]) # (3,)


print("x.shape:", x.shape) # (13, 3)
print("y.shape:", y.shape) # (13,)


# x = x.reshape(4, 3, 1)                      
x = x.reshape(x.shape[0], x.shape[1], 1) # (13, 3, 1)

print(x.shape)


# 2. 모델구성
model = Sequential()
# model.add(LSTM(10, activation='relu', input_shape=(3,1)))
model.add(LSTM(900, input_length=3, input_dim=1))
model.add(Dense(450))
model.add(Dense(230))
model.add(Dense(100))
model.add(Dense(1))

model.summary()


# 3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=800, batch_size=32)

x_predict = x_predict.reshape(1,3,1)


# 4. 예측
print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)