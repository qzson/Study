# 20-06-09_21 화요일 14:00 ~
# Conv1D
# Keras42 Copy
# 시계열 문제를 풀이 시, LSTM은 시간이 많이 걸리므로 대게 Conv1D를 많이 사용한다
# 성능은 90% 이상 유지되며 연산이 LSTM보다 적다
# LSTM과 레이어 구성 형식은 거의 비슷하나 node 옆에 정수가 하나 더 붙는다
# 연습게임은 conv1D로 한번 돌려보고 본모델에선 lstm을 돌린다

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Flatten, MaxPooling1D

# 1. 데이터
a = np.array(range(1, 101))
size = 5                    # timp_steps : 4

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a, size)
print(dataset)
print(dataset.shape)                  # 96,5

x = dataset[:90, 0:4]                 # :90, = 90행까지, 0:4 = 인덱스 3 까지 슬라이싱
y = dataset[:90, 4]                   # :90, = 90행까지, 인덱스 4 부분만 슬라이싱.
x_pred = dataset[-6:, 0:4]            # 마지막 행에서 6번쨰 인덱스, 인덱스 3 까지 슬라이싱
# x_pred = dataset[90:96, 0:4]        # x_pred 같은 결과치

print(x.shape)                            # 90,4
# print(x)
print(y.shape)                            # 90,
# print(y)
print(x_pred.shape)                       # 6,4
# print(x_pred)

x = x.reshape(x.shape[0],x.shape[1],1)    # 90, 4, 1
x_pred = x_pred.reshape(6,4,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state = 66, shuffle = True,
    # x, y, shuffle = False,
    train_size = 0.8)


# 2. 모델
model = Sequential()
# model.add(LSTM(140, input_shape= (4, 1)))
model.add(Conv1D(30, 2, input_shape=(4, 1)))
# model.add(MaxPooling1D())
model.add(Conv1D(50, 2, padding='same'))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(60))  
model.add(Dense(25))
model.add(Dense(1))

model.summary()
# >> (none, 3, 140) 첫 번째 노드 140이 아웃풋으로 들어간다


# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience=100, mode = 'auto')

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics = ['mse'] )
model.fit(x_train, y_train, epochs=500, batch_size=32, verbose=2,
          validation_split=0.2,     # train의 20%
          shuffle=True,             # 셔플 사용 가능
          callbacks=[es])


# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=32)
print('loss:', loss)
print('mse:', mse)

y_predict = model.predict(x_pred)
print(y_predict)