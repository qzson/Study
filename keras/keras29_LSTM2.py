# 20.05.21 0900~

'''
튜닝 결과값 = 7.9774485
'''

from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# import numpy as np
# x = np.array

# 1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])    # (4,3)
y = array([4,5,6,7])                            # (4, ) : 스칼라 4, 벡터 1 // (4, ) != (4,1)
# y2 = array([[4,5,6,7]])                         # (1,4)
# y3 = array([[4],[5],[6],[7]])                   # (4,1)

print("x.shape:", x.shape)                      # (4,3)
print("y.shape:", y.shape)                      # (4, )

# x = x.reshape(4, 3, 1)                      
x = x.reshape(x.shape[0], x.shape[1], 1)

'''             행,          열,    몇 개씩 자르는지
x의 shape = (batch_size, timesteps, feature)
batch_size=n : 데이터 n행씩 짤리는 것 (행의 크기대로 자르는 것) [fit에서]
feature      :
input_shape = (timesteps, feature)
input_length = timesteps / input_dim = feature
timesteps = [1,2,3] / [2,3,4] / ... 하나의 행. (몇 일씩 자르는지)
'''
print(x.shape)


# 2. 모델구성
model = Sequential()
# model.add(LSTM(10, activation='relu', input_shape=(3,1)))
model.add(LSTM(900, input_length=3, input_dim=1))
model.add(Dense(450))
model.add(Dense(230))
model.add(Dense(100))
model.add(Dense(1)) # 1개 예측 y = [4,5,6,7]

model.summary()


# 3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=800, batch_size=32)

x_predict = array([5,6,7])            # 5,6,7 을 통해서 8을 나오게 만들어보고 싶다.
x_predict = x_predict.reshape(1,3,1)    # (3, )


# 4. 예측
print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)