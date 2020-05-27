# 200527 0930~1000

''' keras53 튜닝 값 (0.985 이상)
 loss : 0.08200428349219147
 acc : 0.9776999950408936
 rmsprop
    keras54 튜닝 값 (0.9925 이상) - Dropout 사용
 loss : 0.03950307280309499
 acc : 0.9879000186920166
 rmsprop

 loss : 0.04563406388759613
 acc : 0.9861999750137329
 adam

 gpu epoch 30 / batch 64
 loss : 0.060283861718709524
 acc : 0.9884999990463257
  '''

import numpy as np
import matplotlib.pyplot as plt

# Datasets 불러오기
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train[0])                     # 5의 숫자표현 이미지
print('y_train : ', y_train[0])         # 5

print(x_train.shape)                # (60000, 28, 28)
print(x_test.shape)                 # (10000, 28, 28)
print(y_train.shape)                # (60000,)
print(y_test.shape)                 # (10000,)

print(x_train[0].shape)             # (28, 28) 이미지 사이즈
# plt.imshow(x_train[0], 'gray')
 # plt.imshow(x_train[0]) # 색을 제거 한 것
# plt.show()
 # (28, 28) 짜리가 6만장 // 가로 28픽셀 세로 28픽셀짜리 데이터



# 데이터 전처리 1. OneHotEncording
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)        # (60000, 10)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.layers import Dropout

model = Sequential()
model.add(Conv2D(10, (2,2), activation='relu', padding='same', input_shape=(28,28,1)))
model.add(Dropout(0.2))
model.add(Conv2D(25, (2,2), activation='relu', padding='same'))
model.add(Conv2D(50, (2,2), activation='relu', padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(60, (2,2), activation='relu', padding='same'))
model.add(Conv2D(30, (2,2), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(20, (2,2), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.summary()
# drapout을 한다고 해서 효과가 좋아진다는 보장은 없다.
# 그리고 maxpooling레이어는 드랍아웃이 적용 안된다.


# 3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
 # 2진 분류 : binary_crossentropy / 다중 분류 : categorical_crossentropy
hist = model.fit(x_train, y_train, epochs=30, batch_size=64, verbose=1, validation_split=0.2)

plt.plot(hist.history['loss'])                      # 'loss'값을 y로 넣겠다./ 하나만 쓰면 y 값으로 들어감
plt.plot(hist.history['val_loss'])                  # 시간에 따른 loss, acc여서 x 값으로는 자연스럽게 epoch가 들어감
plt.plot(hist.history['acc']) 
plt.plot(hist.history['val_acc']) 
plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss','val loss','train acc','val acc'])    # 선에 대한 색깔과 설명이 나옴
plt.show()

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=64)
print("loss :", loss)
print("acc :", acc)

# # x_pred = np.array([1,2,3])
# y_pred = model.predict(x_pred)
# # y_pred = np.argmax(y_pred, axis=1)+1
# print(y_pred)
# print(y_pred.shape)