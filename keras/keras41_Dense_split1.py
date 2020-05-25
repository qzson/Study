# 200525 1140~
# Dense 모델을 완성하시오.

''' 튜닝 값
    loss: 1.2505552149377763e-12
 [[4.9999995]
 [6.       ]
 [7.       ]
 [8.000002 ]
 [9.       ]
 [9.999998 ]]
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input

# 1. 데이터
a = np.array(range(1, 11))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i + size)]
        aaa.append([item for item in subset])
    return np.array(aaa)

dataset = split_x(a, size)
print(dataset)
print(dataset.shape)                # 6,5

x = dataset[:, 0:4]                 # : 모든 행, 그 다음 0:4
y = dataset[:, 4]                   # : 모든 행, 인덱스 4부분만 가져오겠다.

print(x.shape)                      # 6,4
print(y.shape)                      # 6,


# 2. 모델
model = Sequential()
model.add(Dense(100, input_dim= 4)) # input_length : time_step (열)
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience=100, mode = 'auto')

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics = ['mse'] )
model.fit(x, y, epochs=800, batch_size=16, verbose=2,
          callbacks=[es])


# 4. 예측
loss, mse = model.evaluate(x, y, batch_size=16)
print('loss:', loss)
print('mse:', mse)

y_predict = model.predict(x)
print(y_predict)