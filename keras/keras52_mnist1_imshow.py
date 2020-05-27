# 200527 0930~1000

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0])
print('y_train : ', y_train[0])
 # 두개의 값은 쌍일 것 F5 실행 확인
 # 각 컬럼마다 0~255의 숫자가 포함되어 있다. 0:하얀색, 255:가장 진한 검정색

print(x_train.shape)                # (60000, 28, 28)
print(x_test.shape)                 # (10000, 28, 28)
print(y_train.shape)                # (60000,)
print(y_test.shape)                 # (10000,)
''' print 값
 정립할 필요성 있음 이해 x
 디멘션은 1개 '''

print(x_train[0].shape)             # (28, 28)
# plt.imshow(x_train[0], 'gray')
 # plt.imshow(x_train[0]) # 색을 제거 한 것
# plt.show()
# (28, 28) 짜리가 6만장 -> imshow가 28바이28 넣은거니까 가로세로 ??넣는다? // 가로 28픽셀 세로 28픽셀짜리 데이터







