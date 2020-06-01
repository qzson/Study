# 20-06-01 / 월 : 1100 ~
# keras90 copy

import numpy as np
import matplotlib.pyplot as plt


# Datasets 불러오기
from keras.datasets import mnist

x_train = np.load('./data/mnist_x_train.npy')
x_test = np.load('./data/mnist_x_test.npy')
y_train = np.load('./data/mnist_y_train.npy')
y_test = np.load('./data/mnist_y_test.npy')


# 데이터 전처리 1. OneHotEncoding
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)        # (60000, 10)

# 데이터 전처리 2. 정규화
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255


# 2. 모델 불러오기 (checkpoint로)
from keras.models import load_model
model = load_model('./model/check-05-0.0424.hdf5')
# (save_wights_only = False)
# model과 weight가 같이 저장되어 있음 
# model, compile, fit부분이 필요없다.


# 4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test, batch_size=128)
print('loss_acc : ', loss_acc)
# loss_acc :  [0.034286081879772244, 0.9884999990463257]
# 결과 바로가 나온다.