# 20-08-05
# Autoencoder noise and CNN (valid)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

# AE 함수 make
def autoencoder():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding = 'valid', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(16, (2, 2), activation='relu', padding='valid'))
    
    model.add(Conv2D(8, (2, 2), activation='relu', padding='valid'))
    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(16, (3, 3), activation='relu', padding='valid'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (2, 2), activation='relu', padding='valid'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(1, (1, 1), activation='sigmoid', padding='same'))
    
    model.summary()
    return model

from tensorflow.keras.datasets import mnist

# mnist 불러오기
train_set, test_set = mnist.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1).astype('float32')/255.
print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)

# 노이즈 추가
x_train_noised = x_train + 0.5 * np.random.normal(0, 0.4, size=x_train.shape)
x_test_noised = x_test + 0.5 * np.random.normal(0, 0.4, size=x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)


model = autoencoder()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# model.compile(optimizer='adam', loss='mse', metrics=['acc'])

model.fit(x_train_noised, x_train, epochs=10, batch_size=32)

output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize=(17, 7))

# 이미지 다섯 개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel('INPUT', size = 25)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 노이즈 있는 친구
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel('NOISE', size = 25)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel('OUTPUT', size = 25)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

# 32node = loss: 0.1660 - acc: 0.8022
# 64node = loss: 0.1353 - acc: 0.8071