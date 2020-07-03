# 20-07-03_28
# regularizer 정규화
# 과적합 피하기 2


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
from keras.layers import Flatten, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2

# kl1 = l1(0.001)
kl2 = l2(0.001)
act = 'elu'

model = Sequential()
model.add(Conv2D(32, kernel_size=3, padding='same', activation=act, input_shape=(32, 32, 3)))
# model.add(Dropout(0.2))
model.add(Conv2D(32, kernel_size=3, padding='same', activation=act))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=3, padding='same', activation=act))
# model.add(Dropout(0.3))
model.add(Conv2D(64, kernel_size=3, padding='same', activation=act))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(128, kernel_size=3, padding='same', activation=act))
# model.add(Dropout(0.4))
model.add(Conv2D(128, kernel_size=3, padding='same', activation=act))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation=act))
model.add(Dropout(0.2))
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

# dp 0.2 통일
loss, acc 는  [0.8025406148910522, 0.7200999855995178]

# elu
loss, acc 는  [0.701026141500473, 0.7631999850273132]

'''