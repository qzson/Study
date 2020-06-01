# 20-06-01 / 월 : 1100 ~
# keras90 copy

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

'''
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
# EarlyStopping, ModelCheckpoint
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'loss', patience = 20)

modelpath = './model/check-{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     verbose=1,
                     save_best_only=True, save_weights_only=False)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train,
                 epochs=5, batch_size=128, verbose=2,
                 validation_split=0.2,
                 callbacks=[es, cp])
'''

from keras.models import load_model
model = load_model('./model/check-05-0.0424.hdf5')
 # checkpoint 최고의 값 불러오기
 # (save_weights_only = False)
 # model과 weight가 같이 저장되어 있음
 # model, compile, fit 부분이 필요없다.


# 4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test, batch_size=128)
print('loss_acc : ', loss_acc)
# loss_acc :  [0.034286081879772244, 0.9884999990463257]

"""
# model, weight 저장 방법들:
1. model.save( '경로/ 파일명' )
  1) fit 전 
    : model 을 저장한다.                 -> compile, fit 필요 O
  2) fit 후 
    : model 과 최종 weight을 저장한다.    -> compile, fit 필요 X
    # from keras.models import load_model
      model = load_model('./model/model_test01.h5')

2. model.save_weights( '경로/ 파일명' ) 
    : 각 레이어의 weight를 저장한다.      -> model, compile 필요 O
                                          : weight를 저장한 model과 구성이 같아야 한다.
                                            -> 저장된 각 레이어의 weight가 model과 매치되어야 하기 때문에  
    # model.load_weights('./model/test_weight1.h5')

3. Modelcheckpoint
   1) save_weights_only = True
    : 각 epoch마다의 weight를 저장한다  
   2)      ,,           = False
    : model과 각 epoch마다의 weight를 저장한다. 
                                        -> model, compile, fit 필요 X
    # from keras.models import load_model
      model = load_model('./model/check-05-0.0424.hdf5') 
>>> model.save / model.save_weight / save_checkpoint 중 뭐가 더 좋은지는 정확하지 않다.
>>> 결과 값(loss, acc)을 보고 더 좋은 것을 사용한다. 
"""