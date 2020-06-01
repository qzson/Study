# 20-06-01 / 월 : 1020 ~
# ==================================================== #
# keras86_load_model1 copy
# 레이어를 3개 추가했을 때?
# 여기서 얻을 수 있는 교훈은 ?
# = 

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


# 2. 모델 구성
from keras.models import load_model
from keras.layers import Dense      # 적용 위해서 추가해줌

model = load_model('./model/model_test01.h5')

# 레이어 3개 추가
model.add(Dense(10, activation='relu', name='add1'))
model.add(Dense(10, activation='relu', name='add2'))
model.add(Dense(10, activation='softmax', name='add3'))
 # 레이어가 추가되어 compile, fit을 다시 한번 해준다.
 # >> loss_acc :  [2.3360459106445313, 0.09650000184774399]
 # keras85 와 달라진 결과값 확인 가능.

model.summary()

# 4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test, batch_size=128)
print('loss_acc : ', loss_acc)

# 예측
y_pred = model.predict(x_test[0:10])
# print(y_pred.shape) # (10, 10)
y_pred = np.argmax(y_pred, axis=1)
print(y_test[0:10])
print(y_pred)