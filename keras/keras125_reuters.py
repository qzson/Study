# 20-07-06_29

from keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. data
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=1000, test_split=0.2)
# 가장 빈도수가 많은 것부터 1000번쨰까지

print(x_train.shape, x_test.shape)  # (8982,) (2246,)
print(y_train.shape, y_test.shape)  # (8982,) (2246,)

# print(x_train[0])
# print(y_train[0])

# print(x_train[0].shape)   에러 : list는 shape구할 수 없다
print(len(x_train[0]))  # 87 (이것들의 크기는 일정하지 않다)

# y의 카테고리 계수 출력
category = np.max(y_train) + 1
print('카테고리 :', category)

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)

y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count()
print(bbb)
print(bbb.shape)

# 주간과제  : groupby()의 사용법 숙지할 것

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, maxlen=100, padding='pre')  # maxlen(최대가 100) / x[0] 문자가 87개, 13개를 0으로 채울 것 / truncating(값을 앞 or 뒤에서 잘라서 날리는 것)
x_test = pad_sequences(x_test, maxlen=100, padding='pre')

# print(len(x_train[0]))
# print(len(x_train[1]))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape)


# 2.모델
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten

model = Sequential()
# model.add(Embedding(1000, 128, input_length=100))
model.add(Embedding(1000, 128))
# parameter 개수가 너무 커져도 좋지 않고 알아서 판단해야함
# word_size 통상적으로 들어가는 단어 개수 넣어주는 것
# output*word_size=첫번째 파라미터 개수
# input_length안 써주면 x_train의  input_length 값을 자연스럽게 가져옴
# Embedding은 3차

model.add(LSTM(64))
model.add(Dense(46, activation='softmax'))

# model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(x_train, y_train, batch_size=100, epochs=10, validation_split=0.2)

acc = model.evaluate(x_test, y_test)[1]
print('\nacc : %.4f' % acc)
# acc : 0.6376
# acc : 0.6273 (인풋 렌스 없이)

y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker='.', c='red', label='TestSet Loss')
plt.plot(y_loss, marker='.', c='blue', label='TrainSet Loss')
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()