# 20-06-01 / 월 : 0900 ~
# ==================================================== #
# keras85 copy

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


# 2. 모델 구성 (Load 방식_keras85_save_model)
from keras.models import load_model

model = load_model('./model/model_test01.h5')
'''
 # 모델과 가중치가 저장이 되었다.
 # save를 complie, fit 부분까지 한다면, 가중치까지 저장되어 빠르게 load 하여 평가 값을 얻을 수 있다.
 >> loss_acc :  [0.03400268962737173, 0.989799976348877]
 keras85 과 같은 evaluate 결과 값이 나온다.

'''


# 4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test, batch_size=128)
print('loss_acc : ', loss_acc)

# 예측
y_pred = model.predict(x_test[0:10])
# print(y_pred.shape) # (10, 10)
y_pred = np.argmax(y_pred, axis=1)
print(y_test[0:10])
print(y_pred)