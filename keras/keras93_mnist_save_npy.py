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

''' NUMPY 로 저장 '''
np.save('./data/mnist_x_train.npy', arr = x_train)
np.save('./data/mnist_x_test.npy', arr = x_test)
np.save('./data/mnist_y_train.npy', arr = y_train)
np.save('./data/mnist_y_test.npy', arr = y_test)
# np.save( '------경로----------', arr = 저장할 내용)


# 데이터 전처리 1. OneHotEncoding
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)        # (60000, 10)

# 데이터 전처리 2. 정규화
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

# 전처리 전에 데이터를 저장하느냐 후에 하느냐는 자유이지만, 이왕이면 전처리 전에 하는 것이 낫다.