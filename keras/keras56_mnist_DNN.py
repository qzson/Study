# 20-05-28_14 0900~
# keras54 pull
# mnist DNN 으로 모델 구성
# acc = 0.98 이상 결과 도출

''' 튜닝 값 (0.98 이상)
 <epoch:225, batch:512>
 loss : 0.15608389074802398
 acc : 0.9801999926567078
  '''

import numpy as np

# Datasets 불러오기
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0])                   # 0 ~ 255까지의 숫자가 적혀짐 (color에 대한 수치)
print('y_train : ', y_train[0])     # 5

print(x_train.shape)                # (60000, 28, 28)
print(x_test.shape)                 # (10000, 28, 28)
print(y_train.shape)                # (60000,)
print(y_test.shape)                 # (10000,)


# 데이터 전처리 1. OneHotEncoding
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)        # (60000, 10)

# 데이터 전처리 2. 정규화
x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype('float32')/255

# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)

# 2. 모델
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

model = Sequential()
model.add(Dense(50, activation='relu', input_dim = 784))
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

# matplotlib 사용
 # import matplotlib.pyplot as plt

 # plt.plot(hist.history['loss'])
 # plt.plot(hist.history['val_loss'])
 # plt.plot(hist.history['acc'])
 # plt.plot(hist.history['val_acc'])
 # plt.title('loss & acc')
 # plt.ylabel('loss, acc')
 # plt.xlabel('epoch')
 # plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
 # plt.show()

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=256)
print("loss :", loss)
print("acc :", acc)