# 200525 1200~
# LSTM 모델을 완성하시오.
# 실습 1. train, test 분리할 것 (8:2)
# 실습 2. 마지막 6개의 행을 predict 로 만들고 싶다. (90행을 test로 잡으면)
# 실습 3. validation 을 넣을 것 (train의 20%)

''' 튜닝값
    loss: 0.0014429715229198337
 [[94.88233 ]
 [95.80324 ]
 [96.706436]
 [97.59027 ]
 [98.453094]
 [99.29344 ]]
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

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
model.add(LSTM(140, input_shape= (4, 1)))
model.add(Dense(100))   
model.add(Dense(60))   
model.add(Dense(25))   
model.add(Dense(1))

model.summary()


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