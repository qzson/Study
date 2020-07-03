# 20-07-03_28
# vgg
# 전이학습?
# 이미지 쪽 이미지 분석할 때 가져다 쓸 수 있다.
# 서머리로 봐서 동일하게 유사하게 구성해도 된다.
# 얘가 이미지넷에서 준우승을 했기 때문에 그와 유사한 데이터셋등이 나오면 활용 가능하다.

from keras.applications import VGG16, VGG19
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation, Dropout
from keras.optimizers import Adam, SGD

from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)        # (50000, 32, 32, 3)
print(x_test.shape)         # (10000, 32, 32, 3)
print(y_train.shape)        # (50000, 1)
print(y_test.shape)         # (10000, 1)

##### 데이터 전처리 2. 리쉐이프 & 정규화
x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255
x_test = x_test.reshape(10000, 32, 32, 3).astype('float32')/255

ishape = (32,32,3)

vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=ishape)   # (None, 224, 224, 3)
# vgg16.summary()

act = 'relu'
model = Sequential()

model.add(vgg16)
model.add(Flatten())
model.add(Dense(256))
# model.add(BatchNormalization())
model.add(Activation(act))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

for layer in model.layers[:19]:
    layer.trainable = False

model.summary()

model.compile(optimizer=SGD(lr=1e-4, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['acc'])

model.fit(x_train, y_train,
          epochs=30, batch_size=32, verbose=1,
          validation_split=0.3)

model.save_weights('param_vgg.hdf5')

scores = model.evaluate(x_test, y_test, verbose=1)
print('test loss :', scores[0])
print('test acc :', scores[1])

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[i])
plt.suptitle('The first ten of the test data', fontsize=16)
plt.show()

pred = np.argmax(model.predict(x_test[0:10]), axis=1)
print(pred)

# test loss : 27.668121783447265
# test acc : 0.07050000131130219
# [0 0 0 5 0 6 1 1 0 5]