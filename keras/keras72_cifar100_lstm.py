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

print(y_train.shape)                # (50000, 100)

##### 데이터 전처리 2. 리쉐이프 & 정규화
x_train = x_train.reshape(-1, 64, 48).astype('float32')/255
x_test = x_test.reshape(-1, 64, 48).astype('float32')/255


##### 2. 모델
from keras.models import Model
from keras.layers import Dense, Input, LSTM
from keras.layers import Dropout

input1 = Input(shape=(64, 48))

lstm1 = LSTM(40, activation='relu', name='dense1')(input1)
dense1 = Dense(500, activation='relu')(lstm1)
dp1 = Dropout(0.2)(dense1)
dense1 = Dense(300, activation='relu')(dp1)
dense1 = Dense(100, activation='relu')(dense1)
dense1 = Dense(50, activation='relu')(dense1)
# dp3 = Dropout(0.2))

output1 = Dense(100, activation='softmax', name='output1')(dense1)
model = Model(inputs=input1, outputs=output1)
model.summary()


##### EarlyStopping & Modelcheckpoint & Tensorboard
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
es = EarlyStopping(monitor='loss', patience=50)

modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto')

tb = TensorBoard(log_dir='graph', histogram_freq=0,
                 write_graph=True, write_images=True)
     # (cmd에서) -> d: -> cd study -> cd graph -> tensorboard --logdir=.
     # 127.0.0.1:6006

##### 3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train,
          epochs=30, batch_size=700, verbose=2,
          validation_split=0.4,
          callbacks=[es, cp, tb])


##### 4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test, batch_size=700)

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

plt.figure(figsize=(10,7))

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
loss, acc 는  [3.4973088026046755, 0.17679999768733978] 
'''