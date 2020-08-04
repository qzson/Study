# 20-08-04 pca (k56과 비교)
# k56_mnist_DNN.py COPY
# pca로 만든 컬럼을 가지고 모델에 넣을 수 있다.
# input_dim = 154로 모델 만드시오
# k56보다 다름


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

# 데이터 전처리 1. OneHotEncoding
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)        # (60000, 10)

# 데이터 전처리 2. 정규화
x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype('float32')/255

X = np.append(x_train, x_test, axis=0)
print(X.shape)  # (70000, 784)

from sklearn.decomposition import PCA

pca = PCA(n_components=154)
pca.fit(X)
X_pca = pca.transform(X)
print(X_pca.shape)  # (70000, 154)

X_train = X_pca[:60000,]
X_test = X_pca[60000:,]
print(X_train.shape)    # (60000, 154)
print(X_test.shape)     # (10000, 154)


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Dense(50, activation='relu', input_dim = 154))
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(200, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

model.summary()

# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience=50, mode = 'auto')

# 3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(X_train, y_train, epochs=100, batch_size=256, verbose=2, validation_split=0.2,
                 callbacks=[es])

# matplotlib 사용
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
plt.show()

# 4. 평가, 예측
loss, acc = model.evaluate(X_test, y_test, batch_size=256)
print("loss :", loss)   # loss : 0.26845426033909897
print("acc :", acc)     # acc : 0.9706