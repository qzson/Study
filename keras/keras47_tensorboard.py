# 200525 1600~
# tensorboard 사용 웹 그래프 표현법


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
a = np.array(range(1, 101))
size = 5

def split_x(seq, size):
    aaa = []        # 는 리스트
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
# print(dataset)
# print(dataset.shape)

x = dataset[:, 0:4]                 # : 모든 행, 그 다음 0:4
y = dataset[:, 4]                   # : 모든 행, 인덱스 4부분만 가져오겠다.

print(x.shape)
print(y.shape)

x = np.reshape(x, (96,4,1))
print(x.shape)


# 2. 모델
# from keras.models import load_model
# model = load_model('./model/save_keras44.h5')
model = Sequential()
model.add(LSTM(5, input_shape=(4,1)))
model.add(Dense(3))
model.add(Dense(1))

model.summary()


# 3. 실행
model.compile(loss='mse', optimizer='adam', metrics = ['acc'] )

# 3-1. 얼리스타핑
from keras.callbacks import EarlyStopping, TensorBoard
tb_hist = TensorBoard(log_dir='graph', histogram_freq=0,    # graph라고 써주고 어떻게 이용되는지 본다.
                      write_graph=True, write_images=True)
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')

# cmd -> d: -> cd study -> cd graph -> tensorboard --logdir=.
# 127.2.0.0 // 6006 포트를 사용하겠다. = 127.0.0.1:6006/

# 3-2. 훈련
hist = model.fit(x, y, epochs=100, batch_size=1, verbose=2,
          validation_split=0.2,
          callbacks=[early_stopping, tb_hist])

# print(hist)                # 자료형만 출력
# print(hist.history.keys()) # dict_keys(['loss', 'mse']) 키 loss와 mse가 있는데 각각 벨류도 있을 것이다.

import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])      # 히스토리에 저장된 애들을 끄집어온다.
plt.plot(hist.history['val_loss'])  # x,y값 둘다 드가거나 y값 집어넣거나
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
# 검증이 실질적으로 좋지 않게 나왔다. (val 그래프가 위로 치고 있으니)
# plt.show() 지금은 필요 x


'''
# 4. 예측
loss, mse = model.evaluate(x, y)
y_predict = model.predict(x)
print('loss:', loss)
print('mse:', mse)
print('y_predict:', y_predict)
'''

# 내일 LSTM dense 분류모델 cnn모델 lstm에서 추가적인 다항적인 lstm 모델 들어갈 수 있다.