# 20-07-07_30
# Bidirected

'''
날씨를 가지고 거꾸로 그 전날도 맞출 수 있다 
데이터 연산이 2배, 방향성도 거꾸로지만, 순서대로 될 것. 즉, 시계열 작업을 2번 하는 것. 그것이 bidirected
lstm 등 시계열 쪽에서 래핑을 해버린다. lstm을 래핑하면, 양방향으로 먹혀주겠다. 라는 뜻. 기본적인 lstm을 더 강력하게 만들어버린다
'''

from keras.datasets import reuters, imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=2000)
print(x_train.shape, x_test.shape)  # (25000,) (25000,)
print(y_train.shape, y_test.shape)  # (25000,) (25000,)

# print(x_train[0])
# print(y_train[0])

# print(x_train[0].shape)   에러 : list는 shape구할 수 없다
print(len(x_train[0]))                  # 218 (이것들의 크기는 일정하지 않다)

# y의 카테고리 계수 출력
category = np.max(y_train) + 1
print('카테고리 :', category)            # 2

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)                          # [0 1]

y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count()
print(bbb)                              # 0 : 12500 , 1 : 12500
print(bbb.shape)                        # (2,)


from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, maxlen=111, padding='pre')  # maxlen(최대가 100) / x[0] 문자가 87개, 13개를 0으로 채울 것
x_test = pad_sequences(x_test, maxlen=111, padding='pre')    # truncating(값을 앞 or 뒤에서 잘라서 날리는 것)

print(len(x_train[0]))
print(len(x_train[1]))

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape)  # (25000, 111) (25000, 111)


# 2.모델
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten
from keras.layers import Conv1D, Dropout, MaxPooling1D, Bidirectional

model = Sequential()
# model.add(Embedding(2000, 128, input_length=111))
model.add(Embedding(2000, 128))

model.add(Conv1D(10, 5, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(pool_size=4))

model.add(Bidirectional(LSTM(10)))
# lstm을 두번하는 것과 차이는?
# bid : 정방향 갔다 역방향 갔다. 가중치를 초기화 ?


model.add(LSTM(10))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# history = model.fit(x_train, y_train, batch_size=100, epochs=10, validation_split=0.2)

# acc = model.evaluate(x_test, y_test)[1]
# print('\nacc : %.4f' % acc)
# # acc : 0.8352 // loss: 0.1733 - acc: 0.9314 - val_loss: 0.5241 - val_acc: 0.8264 <num_word : 2000>
# # acc : 0.8312 // loss: 0.2654 - acc: 0.8870 - val_loss: 0.4019 - val_acc: 0.8290 <num_word : 1000>

# y_val_loss = history.history['val_loss']
# y_loss = history.history['loss']

# plt.plot(y_val_loss, marker='.', c='red', label='TestSet Loss')
# plt.plot(y_loss, marker='.', c='blue', label='TrainSet Loss')
# plt.legend(loc='upper right')
# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()

# # 1. imdb 검색해서 데이터 내용 확인
# # 2. word_size 전체데이터 부분 변경해서 최상값 확인
# # 3. 주간과제 : groupby()의 사용법 숙지할 것
# # 4. 인덱스를 단어로 바꿔주는 함수 찾을 것
# # 5. 125번, 126번 튠