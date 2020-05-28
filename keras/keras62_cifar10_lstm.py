# 20-05-28 14:38~
'''
 category
 loss : 1.4465326571464538
 acc : 0.46700000762939453
 '''
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
print(y_train.shape)                # (50000, 10)

# 데이터 전처리 2. 리쉐이프 & 정규화
x_train = x_train.reshape(-1, 64, 48).astype('float32')/255
x_test = x_test.reshape(-1, 64, 48).astype('float32')/255

# 2. 모델

input1 = Input(shape=(64, 48))

lstm1 = LSTM(40, activation='relu', name='dense1')(input1)
dense1 = Dense(500, activation='relu')(lstm1)
dp1 = Dropout(0.2)(dense1)
dense1 = Dense(300, activation='relu')(dp1)
dense1 = Dense(100, activation='relu')(dense1)
dense1 = Dense(50, activation='relu')(dense1)
# dp3 = Dropout(0.2))

output1 = Dense(10, activation='softmax', name='output1')(dense1)

model = Model(inputs=input1, outputs=output1)

model.summary()

# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=50, mode='auto')

# 3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=30, batch_size=700, verbose=1, validation_split=0.2,
          callbacks=[es])


# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=700)
print("loss :", loss)
print("acc :", acc)
