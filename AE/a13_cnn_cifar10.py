# 20-08-05
# Autoencoder CNN and cifar10

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, Activation

# AE 함수 make
def autoencoder():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding = 'valid', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(128, (3, 3), activation='relu', padding='valid'))
    model.add(Conv2DTranspose(64, (3, 3), activation='relu', padding='valid'))
    model.add(Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='valid'))
    
    model.summary()
    return model

from tensorflow.keras.datasets import cifar10

# cifar10 불러오기
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)

# # cifar10 레이블 1 car 이미지 만
# x_train = x_train[np.where(y_train==1)[0],:,:,:]
# x_test = x_test[np.where(y_test==1)[0],:,:,:]
# print(x_train.shape, x_test.shape)  # (5000, 32, 32, 3) (1000, 32, 32, 3)

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.


model = autoencoder()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# model.compile(optimizer='adam', loss='mse', metrics=['acc'])

model.fit(x_train, x_train, epochs=30, batch_size=128, shuffle=True, validation_data=(x_test, x_test))

output = model.predict(x_test)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(17, 7))

# 이미지 다섯 개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(32, 32, 3), cmap='gray')
    if i == 0:
        ax.set_ylabel('INPUT', size = 25)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(32, 32, 3), cmap='gray')
    if i == 0:
        ax.set_ylabel('OUTPUT', size = 25)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

# loss: 0.5479 - acc: 0.0121 - val_loss: 0.5489 - val_acc: 0.0121