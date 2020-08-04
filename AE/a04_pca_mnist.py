# 20-08-04 Autoencoder's pca

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

X = np.append(x_train, x_test, axis=0)

print(X.shape)  # (70000, 784)

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)
best_n_components = np.argmax(cumsum >= 0.95) + 1
print(best_n_components) # 154
# 154개 정도 주었을 때, 95% 이상의 특징 차원축소 진행
# feature importance 측면에서 어느정도 해야할지 판단이 이제 가능하다.