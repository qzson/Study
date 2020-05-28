# 과제 2 : fashion.dnn
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

# 데이터 전처리 2. 정규화
x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255


# 2. 모델
model = Sequential()
model.add(Dense(50, activation='relu', input_shape = (28*28, )))
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(200, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

model.summary()


# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience=50, mode = 'auto')

# 3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=100, batch_size=256, verbose=2, validation_split=0.2,
                 callbacks=[es])


# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=256)
print("loss :", loss)
print("acc :", acc)

''' 200528 결과값
 loss : 0.7194221629500389
 acc  : 0.8799999952316284
 '''