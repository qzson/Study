# 20-07-03_28
# regularizer 정규화
# 과적합 피하기 3
'''
계단 함수 다음에 나온 것이 sigmoid
https://da-nika.tistory.com/43

그 다음 relu (하지만 음수에 대한 상계 문제) -> 그래서 leakyReLu
그렇지만, 또 다시 상계되는 문제가 생김 -> elu (-1,2,3 음수에 대해서 딱 제한을 걸어 놓았다.)
이것에서 0과 1사이로 밀어 넣는 것이 생겼는데 그것이 selu
규제나 제한하는 것에 대해서 발전했다고 더 좋은 것은 아니다. (데이터에 맞춰 사용해야한다)
>> 하이퍼 파라미터 튜닝 자동화로 그것을 어느정도 맞춰줄 수 있다.

batchnormalization도 역시 레이어에 있는 것들을 normalization 하는 것이다.
이것은 활성화 함수 이전에 해야한다.
활성화함수 통과한 값을 BN를 시키는데 원래 의도는 act 전에 해서 그 일반화 값을 전달해주는 것이 목적

>>> 이런 과적합 피하기 1~3 기능을 다 쓸 수 있는데 그렇다고 성능이 좋아지는 것은 아니다.
>>> 어쩌면 안쓰느니 못한 결과가 나올 수도 있다.
'''


##### 데이터 LOAD
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train[0])
# print('y_train[0] : ', y_train[0])  # [19]

print(x_train.shape)        # (50000, 32, 32, 3)
print(x_test.shape)         # (10000, 32, 32, 3)
print(y_train.shape)        # (50000, 1)
print(y_test.shape)         # (10000, 1)

# plt.imshow(x_train[0])
# plt.show()


##### 데이터 전처리 1. OneHotEncoding
# from keras.utils import np_utils

# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

# print(y_train.shape)                 # (50000, 100)

##### 데이터 전처리 2. 리쉐이프 & 정규화
x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255
x_test = x_test.reshape(10000, 32, 32, 3).astype('float32')/255


##### 2. 모델
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2

# kl1 = l1(0.001)
kl2 = l2(0.001)

model = Sequential()
model.add(Conv2D(32, kernel_size=3, padding='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

model.add(Conv2D(64, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

model.add(Conv2D(128, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['acc'])
                        # 1e-4 = 0.0001 (learning_rate)
                              # loss= 이것은 원핫인코딩 안했을 때 하면 된다. (개인적인 취향) - 어쨌든 원핫인코딩은 해야한다 (다른방법임)
hist = model.fit(x_train, y_train,
          epochs=30, batch_size=128, verbose=1,
          validation_split=0.3)

loss = model.evaluate(x_test, y_test)


##### 4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test, batch_size=32)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

# print('acc 는 ', acc)
# print('val_acc 는 ', val_acc)

# evaluate 종속 결과
print('loss, acc 는 ', loss_acc)


##### plt 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(9,5))

# 1
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

# 2
plt.subplot(2,1,2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()

''' 
loss, acc 는  [1.636397636318207, 0.7167999744415283]

그래프를 보면 전형적인 과적합 
에포 5지점 정도 밖에 
(과적합을 줄이자)

# L2 적용
loss, acc 는  [1.2808144750595092, 0.7092999815940857]
결론 l2 적용했는데, 효과는 조금 있었다.

# dropout
loss, acc 는  [0.8241934020996093, 0.7098000049591064]



'''