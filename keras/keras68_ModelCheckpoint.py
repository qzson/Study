# 20-05-29 / 1000~ 
# k67 copy

''' keras53_mnist 기준
 model.fit 무얼 반환했는가?
 - (fit에 대한 반환 값) : loss, acc(matrics 값)
 - 
  '''


''' 튜닝 값 (0.985 이상)
 keras54_dropout_cnn
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

# EarlyStopping
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'loss', patience = 20)

# ModelCheckpoint
modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
 # 이 경로의 파일명을 modelpath로 하겠다
 # {epoch:02d}-{val_loss:.4f} : epoch를 2자리 정수로 해주고 , val_loss를 4자리 숫자의 float를 쓰겠다
cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',
                     save_best_only = True, mode = 'auto')
                     # save_best_only 좋은 것만 저장을 하겠다.
                     # 얼리스타핑 이전의 좋은 값을 알 수 있다?
                     # 내려가다가 올라가는 지점은 출력을 안해준다

# 3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train,
                 epochs=30, batch_size=64, verbose=1,
                 validation_split=0.2,
                 callbacks=[es, cp])


# 4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test, batch_size=64)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print('acc : ', acc)
print('val_acc : ', val_acc)
print('loss_acc : ', loss_acc)        # evaluate 결과 - 실제로 훈련시키지 않은 데이터를 집어 넣어 나온 결과


import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
 # 10,6 인치 사이즈

plt.subplot(2, 1, 1)
 # (2행, 1열, 1) : 2행 1열의 첫번째 껏 그림을 그리겠다.
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
 # 가로세로 줄을 그어준 모양 넣어준다.
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['loss', 'val_loss'])
plt.legend(loc='upper right')
 # loc = 위치

# 2. (2,1,2)
plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()