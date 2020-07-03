# 20-07-03_28
# regularizer 정규화
# 과적합 피하기 1.

'''
과적합의 이유는 뭘까 ?
- 연산한 값이 튀는 경우가 생긴다 = 그라디언트 폭발, 그라디언트 소실
> 가중치값 자체가 너무 커지게 되면, w = 100만 인데 relu를 만나면 그대로 위로 간다
그리고 그 다음 레이어에 노드에 * 가 된다. 그럼 그 노드 값과 또 떡상하게 된다. 그러면 가중치 하나만으로 컨트롤이 안된다.
>> 활성화 함수가 어느정도 제어를 하지만 그것으로는 부족하다.

- 초기에 최초로 나온 것이 시그모이드 >> 그렇지만 하나가지고는 해결이 안된다.
(가중치 연산.. 계속이어지면 엄청난 큰 값이 되기 때문)
>> 그래서 이것을 제어하기 위해 regularizer가 생겼다.

그러므로 <L1 규제> 와 <L2 규제>를 찾아라
L1 규제 : 가중치의 절대값 합 : regularizer.l1(l=0.01)
L2 규제 : 가중치의 제곱 합 : regularizer.l2(l=0.01)

loss = L1 * reduce_sum(abs(x))
loss = L2 * reduce_sum(square(x))
>> 다음 레이어로 전달되는 loss 혹은 다른 값들을 축소하겠다. 이런 의미.

kernel_regularizer=l2(0.001)
레이어당 곱하기를 한번 더 하기 때문에 속도가 조금 더 느려진다.
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
from keras.layers import Flatten, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2

# kl1 = l1(0.001)
kl2 = l2(0.001)

model = Sequential()
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', kernel_regularizer=kl2))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=kl2))
model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=kl2))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu', kernel_regularizer=kl2))
model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu', kernel_regularizer=kl2))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_regularizer=kl2))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['acc'])
                        # 1e-4 = 0.0001 (learning_rate)
                              # loss= 이것은 원핫인코딩 안했을 때 하면 된다. (개인적인 취향) - 어쨌든 원핫인코딩은 해야한다 (다른방법임)
hist = model.fit(x_train, y_train,
          epochs=20, batch_size=128, verbose=1,
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
'''