# 200525 1500~
# matplotlib 사용 그래프 표현

''' 튜닝 값
    loss: 0.03446273133158684
 [[4.960638 ]
 [6.034794 ]
 [7.0374126]
 [7.9646797]
 [8.815494 ]
 [9.590944 ]]
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

""" 가중치(W) 저장 방법 """
# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics = ['acc'] )
hist = model.fit(x, y, epochs=800, batch_size=32, verbose=2,
                 validation_split=0.2,
                 callbacks=[es])
# hist = model.fit에 훈련시키고 난 loss, metrics안에 있는 값들을 반환한다.

print(hist)   #자료형 모양 <keras.callbacks.callbacks.History object at 0x00000178F0734EC8> : 원래 안보여줌
print(hist.history.keys())                 # dict_keys(['loss', 'mse'])
 
# 그래프 표현법
import matplotlib.pyplot as plt            # 그래프 그리는 것

plt.plot(hist.history['loss'])             # 'loss'값을 y로 넣겠다./ 하나만 쓰면 y 값으로 들어감
plt.plot(hist.history['val_loss'])         # 시간에 따른 loss, acc여서 x 값으로는 자연스럽게 epoch가 들어감
plt.plot(hist.history['acc']) 
plt.plot(hist.history['val_acc']) 
plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss','val loss','train acc','val acc'])
plt.show()
# 검증이 실질적으로 좋지 않게 나왔다. (val 그래프가 위로 치고 있으니)


# 4. 평가, 예측
loss, mse = model.evaluate(x, y, batch_size=32)
print('loss:', loss)
# print('acc:', acc)

y_predict = model.predict(x)
print(y_predict)