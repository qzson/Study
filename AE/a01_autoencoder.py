# 20-08-04 Autoencoder
# k56_mnist_DNN.py COPY


import numpy as np

# Datasets 불러오기
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train[0])                   # 0 ~ 255까지의 숫자가 적혀짐 (color에 대한 수치)
print('y_train : ', y_train[0])     # 5

print(x_train.shape)                # (60000, 28, 28)
print(x_test.shape)                 # (10000, 28, 28)
print(y_train.shape)                # (60000,)
print(y_test.shape)                 # (10000,)


# # 데이터 전처리 1. OneHotEncoding
# from keras.utils import np_utils
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)        # (60000, 10)

# 데이터 전처리 2. 정규화
x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype('float32')/255

# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)

# 2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape=(784,))
encoded = Dense(32, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)     # 전처리 해줘서 시그모이드 (0과 1로 전처리가 되어 값을 만들어 내는데 쓴다)

autoencoder = Model(input_img, decoded)

autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')   # binary를 한다고 해서 무조건 0과 1로 나오는 것은 아니다.
# autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, validation_split=0.2)
decoded_imgs = autoencoder.predict(x_test)

# matplotlib 사용
import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(28, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

plt.show()