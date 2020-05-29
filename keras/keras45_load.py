# 200525 1400~
# save & load

''' 튜닝 값
    loss: 1.9986160623375326e-05
 [[5.002454 ]
 [5.9954915]
 [6.9975085]
 [8.003888 ]
 [9.006316 ]
 [9.994312 ]]
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
a = np.array(range(1, 11))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)                # 6,5
print(dataset)
print(dataset.shape)                      # 6,5

x = dataset[:, 0:4]                       # : 모든 행, 그 다음 0:4
y = dataset[:, 4]                         # : 모든 행, 인덱스 4부분만 가져오겠다.

print(x.shape)                            # 6,4
print(y.shape)                            # 6,

x = x.reshape(x.shape[0], x.shape[1], 1)
# x = np.reshape(x, (6,4,1))
# x = x.reshape(6, 4, 1) 같은 문법
print(x.shape)                            # 6,4,1


# 2. 모델 (저장한 모델 불러오기)
from keras.models import load_model
model = load_model('./model/save_keras44.h5')

model.add(Dense(5, name='dense_x1'))
model.add(Dense(1, name='dense_x2'))

model.summary()


# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=100, mode='auto')

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics = ['mse'] )
model.fit(x, y, epochs=800, batch_size=32, verbose=2,
          callbacks=[es])

# 4. 평가, 예측
loss, mse = model.evaluate(x, y, batch_size=32)
print('loss:', loss)
print('mse:', mse)

y_predict = model.predict(x)
print(y_predict)