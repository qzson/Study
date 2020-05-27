# 200527 0930~1000

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist                          # keras에서 제공되는 예제 파일 

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # mnist에서 이미 x_train, y_train으로 나눠져 있는 값 가져오기

print(x_train[0])                                         # 0 ~ 255까지의 숫자가 적혀짐 (그림)
print('y_train : ', y_train[0])                           # 5
 # 두개의 값은 쌍일 것. F5 실행 확인
 # 각 컬럼 마다 0~255의 숫자가 포함되어 있다. 0:하얀색, 255:가장 진한 검정색

print(x_train.shape)                # (60000, 28, 28)
print(x_test.shape)                 # (10000, 28, 28)
print(y_train.shape)                # (60000,)       : 10000개의 xcalar를 가진 vector(1차원)
print(y_test.shape)                 # (10000,)

print(x_train[0].shape)             # (28, 28)
plt.imshow(x_train[59999], 'gray')  # '2차원'을 집어넣어주면 수치화된 것을 이미지로 볼 수 있도록 해줌
 # plt.imshow(x_train[0]) # 색을 제거 한 것
 # plt.imshow를 두번 쓸 경우 맨 나중에 쓴 코드의 그림만 보여진다.

plt.show()
 # (28, 28) 짜리가 6만장 // 가로 28픽셀 세로 28픽셀짜리 데이터







