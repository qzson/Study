# 200525 1200~
# 42번 카피 Dense로 리뉴얼

''' 튜닝값
    loss: 5.0994861111597345e-11
 [[ 94.999985]
 [ 95.99997 ]
 [ 97.      ]
 [ 98.      ]
 [ 99.000015]
 [100.      ]]
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input

# 1. 데이터
a = np.array(range(1, 101))
size = 5                    # timp_steps : 4

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    return np.array(aaa)

dataset = split_x(a, size)          # 96,5
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

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state = 66, shuffle = True,
    # x, y, shuffle = False,
    train_size = 0.8)

''' 데이터 나누는 다른 방법?
    train, test, predict 값 나누기 : train_test_split이용
    # x, y 나누기
    x = dataset[:, 0:4]
    y = dataset[:, 4]

    # x_predict 값
    from sklearn.model_selection import train_test_split
    x1, x_predict, y1, y_predict = train_test_split(x, y, train_size = 90/96)

    # train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x1, y1, train_size = 0.8)

    -----------------------------------------------------------------------------
    # 두 개의 성능은 동일하다.
    # 다만, slicing은 원론적으로 알 수 있고, 
    #       train_test_split는 percentage(%)로 나눌 수 있어서 좀 더 편리하다.

    print(x_train.shape)
    print(x_test.shape)
    print(x_predict.shape)
'''


# 2. 모델
model = Sequential()
model.add(Dense(100, input_shape= (4, )))                # input_length : time_step (열)
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
model.fit(x_train, y_train, epochs=700, batch_size=32, verbose=2,
          validation_split=0.2,     # train의 20%
          shuffle=True,             # 셔플 사용 가능
          callbacks=[es])


# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=32)
print('loss:', loss)
print('mse:', mse)

y_predict = model.predict(x_pred)
print(y_predict)