# 20-06-01 / 월 : 1100 ~
# keras85 copy
# 체크포인트 save 법

import numpy as np
import matplotlib.pyplot as plt

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
from keras.layers import Conv2D, Dense
from keras.layers import Dropout, MaxPooling2D, Flatten

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


# 3. 훈련
# EarlyStopping, Modelcheckpoint
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'loss', patience = 20)

modelpath = './model/check-{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     verbose=1,
                     save_best_only=True, save_weights_only=False)
                                          # 가중치만 저장하겠다 : False = ( model도 저장 )

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train,
          epochs=5, batch_size=128, verbose=2,
          validation_split=0.2,
          callbacks=[es, cp])
           # 훈련 중에 나오는 가중치 저장
           # >> save_weight와 결과값을 비교하여 더 좋은 것을 사용한다.


# 4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test, batch_size=128)
print('loss_acc : ', loss_acc)