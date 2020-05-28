# 20-05-28 14:38~
# acc 75% 이상

'''
 loss : 0.8723045469284058
 acc : 0.7052000164985657 '''

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train[0])
print('y_train[0] : ', y_train[0])    # [6]

print(x_train.shape)                  # (50000, 32, 32, 3)
print(x_test.shape)                   # (10000, 32, 32, 3)
print(y_train.shape)                  # (50000, 1)
print(y_test.shape)                   # (10000, 1)

plt.imshow(x_train[0])
# plt.show()

# 데이터 전처리 1. OneHotEncoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)                        # (50000, 10)

# 데이터 전처리 2. 정규화
x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255
x_test = x_test.reshape(10000, 32, 32, 3).astype('float32')/255


# 모델
input1 = Input(shape=(32,32,3))

Conv1 = Conv2D(10, (2,2), activation='relu', padding='same', name='Conv1')(input1)
Conv1 = Conv2D(40, (2,2), activation='relu', padding='same', name='Conv2')(Conv1)
Conv1 = Conv2D(70, (2,2), activation='relu', padding='same', name='Conv3')(Conv1)
dp1 = Dropout(0.2)(Conv1)
Conv1 = Conv2D(50, (2,2), activation='relu', padding='same', name='Conv4')(dp1)
Conv1 = Conv2D(40, (2,2), activation='relu', padding='same', name='Conv5')(Conv1)
mp1 = MaxPooling2D(2,2)(Conv1)
Conv1 = Conv2D(30, (2,2), activation='relu', padding='same', name='Conv6')(mp1)
Conv1 = Conv2D(20, (2,2), activation='relu', padding='same', name='Conv7')(Conv1)
Conv1 = Conv2D(10, (2,2), activation='relu', padding='same', name='Conv8')(Conv1)
dp2 = Dropout(0.2)(Conv1)
Conv1 = Conv2D(10, (2,2), activation='relu', padding='same', name='Conv9')(dp2)
ft1 = Flatten()(Conv1)

output1 = Dense(10, activation='softmax', name='output1')(ft1)

model = Model(inputs=input1, outputs=output1)

model.summary()

# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=50, mode='auto')

# 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=30, batch_size=64, verbose=1, validation_split=0.2,
          callbacks=[es])


# 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=64)
print("loss :", loss)
print("acc :", acc)