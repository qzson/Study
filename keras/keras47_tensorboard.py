# 200525 1600~
# tensorboard 사용 웹 그래프 표현법
# lstm 구성


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
a = np.array(range(1, 101))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print(dataset)
print(dataset.shape)                # (96, 5)

x = dataset[:, 0:4]                 # : 모든 행, 그 다음 0:4
y = dataset[:, 4]                   # : 모든 행, 인덱스 4부분만 가져오겠다.

print(x.shape)                      # (96, 4)
print(y.shape)                      # (96, )
print(y)

x = np.reshape(x, (96,4,1))
print(x.shape)                      # (96, 4, 1)


# 2. 모델
# from keras.models import load_model
# model = load_model('./model/save_keras44.h5')
model = Sequential()
model.add(LSTM(5, input_shape=(4,1)))
model.add(Dense(3))
model.add(Dense(1))

model.summary()


# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=5, mode='auto')

# 3. 실행
model.compile(loss='mse', optimizer='adam', metrics = ['acc'] )

# Tensorboard
from keras.callbacks import TensorBoard
tb_hist = TensorBoard(log_dir='graph', histogram_freq=0,    # graph라고 써주고 어떻게 이용되는지 본다.
                      write_graph=True, write_images=True)  # log_dir=' 폴더 ' : 제일 많이 틀림
 # (cmd에서) -> d: -> cd study -> cd graph -> tensorboard --logdir=.
 # 127.2.0.0 // 6006 포트를 사용하겠다. = 127.0.0.1:6006 이것을 웹에 붙여넣기해서 이동

# 3. 훈련
hist = model.fit(x, y, epochs=100, batch_size=1, verbose=2,
          validation_split=0.2,
          callbacks=[es, tb_hist])
 # Tensorboard 사용
 # hist = model.fit에 훈련시키고 난 loss, metrics안에 있는 값들을 반환한다.


# print(hist)                # 자료형만 출력
# print(hist.history.keys()) # dict_keys(['loss', 'mse']) 키 loss와 mse가 있는데 각각 벨류도 있을 것이다.

import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])       # 'loss'값을 y로 넣겠다./ 인자 하나만 쓰면 y 값으로 들어감
plt.plot(hist.history['val_loss'])   # 시간에 따른 loss, acc여서 x 값으로는 자연스럽게 epoch가 들어감
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])     # 선에 대한 색깔과 설명이 나옴
# 검증이 실질적으로 좋지 않게 나왔다. (val 그래프가 위로 치고 있으니)
# plt.show()


'''
# 4. 예측
loss, mse = model.evaluate(x, y)
y_predict = model.predict(x)
print('loss:', loss)
print('mse:', mse)
print('y_predict:', y_predict)
'''

# 내일 LSTM dense 분류모델 cnn모델 lstm에서 추가적인 다항적인 lstm 모델 들어갈 수 있다.