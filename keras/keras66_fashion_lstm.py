# 과제 3 : fashion.lstm
# Sequential 모델 구성
# 하단에 loss, acc

import numpy as np
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train[0])
print('y_train[0] : ', y_train[0])  # 9

print(x_train.shape)                # (60000, 28, 28)
print(x_test.shape)                 # (10000, 28, 28)
print(y_train.shape)                # (60000,)
print(y_test.shape)                 # (10000,)

plt.imshow(x_train[0])
# plt.show()

# 데이터 전처리 1. OneHotEncoding
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)                # (60000, 10)

# 데이터 전처리 2. 리쉐이프 & 정규화
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# 2. 모델
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

model = Sequential()

model.add(LSTM(10, activation='relu', input_shape=(28,28)))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=50, mode='auto')

# 3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
model.fit(x_train, y_train, epochs=50, batch_size=128, verbose=1, validation_split=0.3,
          callbacks=[])


# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=128)
print("loss :", loss)
print("acc :", acc)

''' 200528 결과값
 loss : 0.606213851737976
 acc  : 0.7723000049591064
 '''