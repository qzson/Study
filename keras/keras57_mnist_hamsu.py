# 200528 1000~
# keras54 pull
# 54를 함수형으로 구성해보자

''' 튜닝 값
 loss : 0.028865077007260334
 acc : 0.9919000267982483 '''

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
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

# 2. 모델
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.layers import Dropout, Input

input1 = Input(shape=(28,28,1))

Conv1 = Conv2D(10, (2,2), activation='relu', padding='same', name='Conv1')(input1)
Conv1 = Conv2D(40, (2,2), activation='relu', padding='same', name='Conv2')(Conv1)
Conv1 = Conv2D(70, (2,2), activation='relu', padding='same', name='Conv3')(Conv1)
dp1 = Dropout(0.2)(Conv1)
Conv1 = Conv2D(50, (2,2), activation='relu', padding='same', name='Conv4')(dp1)
Conv1 = Conv2D(40, (2,2), activation='relu', padding='same', name='Conv5')(Conv1)
mp1 = MaxPooling2D(2,2)(Conv1)
Conv1 = Conv2D(30, (2,2), activation='relu', padding='same', name='Conv6')(mp1)
Conv1 = Conv2D(20, (2,2), activation='relu', padding='same', name='Conv7')(Conv1)
Conv1 = Conv2D(10, (2,2), activation='relu', padding='same', name='Conv8')(Conv1)
dp2 = Dropout(0.2)(Conv1)
Conv1 = Conv2D(10, (2,2), activation='relu', padding='same', name='Conv9')(dp2)
ft1 = Flatten()(Conv1)

output1 = Dense(10, activation='softmax', name='output1')(ft1)

model = Model(inputs=input1, outputs=output1)

model.summary()


# 3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=30, batch_size=64, verbose=1, validation_split=0.2)


# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=64)
print("loss :", loss)
print("acc :", acc)
