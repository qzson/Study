# 20-05-29 / 1040 ~
# 함수형 모델 구성
# es, cp, tensorboard, plt 등 시각화 전부 사용

##### 데이터 LOAD
from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train[0])
print('y_train[0] : ', y_train[0])  # [19]

print(x_train.shape)        # (50000, 32, 32, 3)
print(x_test.shape)         # (10000, 32, 32, 3)
print(y_train.shape)        # (50000, 1)
print(y_test.shape)         # (10000, 1)

# plt.imshow(x_train[0])
# plt.show()


##### 데이터 전처리 1. OneHotEncoding
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)                 # (50000, 100)

##### 데이터 전처리 2. 리쉐이프 & 정규화
x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255
x_test = x_test.reshape(10000, 32, 32, 3).astype('float32')/255


##### 2. 모델
from keras.models import Model
from keras.layers import Dense, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout

input1 = Input(shape=(32,32,3))

cv1 = Conv2D(10, (2,2), name='conv1')(input1)
cv1 = Conv2D(40, (2,2))(cv1)
cv1 = Conv2D(70, (2,2))(cv1)
dp1 = Dropout(0.2)(cv1)
cv1 = Conv2D(50, (2,2))(dp1)
cv1 = Conv2D(40, (2,2))(cv1)
mp1 = MaxPooling2D(2,2)(cv1)
cv1 = Conv2D(30, (2,2))(mp1)
cv1 = Conv2D(20, (2,2))(cv1)
cv1 = Conv2D(10, (2,2))(cv1)
dp2 = Dropout(0.2)(cv1)
cv1 = Conv2D(10, (2,2))(dp2)
ft1 = Flatten()(cv1)

output1 = Dense(100, activation='softmax', name='output1')(ft1)
model = Model(inputs=input1, outputs=output1)
model.summary()


##### EarlyStopping & Modelcheckpoint & Tensorboard
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
es = EarlyStopping(monitor='acc', patience=20)

modelpath = './model/{epoch:02d}-{acc:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='acc',
                     save_best_only=True, mode='auto')

tb = TensorBoard(log_dir='graph', histogram_freq=0,
                 write_graph=True, write_images=True)
     # (cmd에서) -> d: -> cd study -> cd graph -> tensorboard --logdir=.
     # 127.0.0.1:6006

##### 3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train,
          epochs=30, batch_size=128, verbose=1,
          validation_split=0.4,
          callbacks=[es, cp, tb])


##### 4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test, batch_size=128)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print('acc 는 ', acc)
print('val_acc 는 ', val_acc)

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

''' ㅅㅂ
# loss, acc 는  [3.433340796661377, 0.2328999936580658]
'''