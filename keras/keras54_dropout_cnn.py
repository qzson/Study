# 200527 53번 연장 + Dropout 적용

''' keras53 튜닝 값 (0.985 이상)
 loss : 0.08200428349219147
 acc : 0.9776999950408936
 rmsprop
    keras54 튜닝 값 (0.9925 이상) - Dropout 사용
 loss : 0.03950307280309499
 acc : 0.9879000186920166
 rmsprop

 gpu epoch 30 / batch 64
 loss : 0.031908373312254844
 acc : 0.9923999905586243 '''

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
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.layers import Dropout

model = Sequential()
model.add(Conv2D(10, (2,2), activation='relu', padding='same', input_shape=(28,28,1)))
model.add(Conv2D(40, (2,2), activation='relu', padding='same'))
model.add(Conv2D(70, (2,2), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(50, (2,2), activation='relu', padding='same'))
model.add(Conv2D(40, (2,2), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(30, (2,2), activation='relu', padding='same'))
model.add(Conv2D(20, (2,2), activation='relu', padding='same'))
model.add(Conv2D(10, (2,2), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(10, (2,2), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.summary()
# Droppout을 한다고 해서 효과가 좋아진다는 보장은 없다.
# 그리고 maxpooling 레이어는 Dropout 이 적용 안된다.


# 3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=30, batch_size=64, verbose=1, validation_split=0.2)


# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=64)
print("loss :", loss)
print("acc :", acc)