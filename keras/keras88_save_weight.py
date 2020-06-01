# 20-06-01 / 월 : 1030 ~
# ==================================================== #
# keras85 copy
# save_weights 방법


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

# """ model 저장 """
# model.save('./model/model_test01.h5')
# # 모델까지만 저장된다.(가중치는 저장이 안된다 / compile, fit 필요 )


# 3. 훈련
# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience = 20)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train,
                 epochs=5, batch_size=128, verbose=2,
                 validation_split=0.2,
                 callbacks=[es])

""" model 저장 """
model.save('./model/model_test01.h5')
 # fit한 다음에 저장하면 '가중치'가 저장된다.(compile, fit할 필요 X)

""" save_weights """
model.save_weights('./model/test_weight1.h5')
 # 각각의 레이어의 weight 저장 (모델 저장 X)


# 4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test, batch_size=128)
print('loss_acc : ', loss_acc)

''' keras85_save 결과 값
loss_acc :  [0.03400268962737173, 0.989799976348877]'''