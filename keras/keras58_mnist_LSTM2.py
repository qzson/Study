# 0528 1117~
# keras54 pull
# 다중 분류모델을 LSTM 으로 구성하기 (0.95 이상) - 차원 낮춰야 한다.
# k54(CNN), k56(DNN), k58(LSTM) 각 비교 및 최적화
#### INPUT : (28, 28) ####

''' <input_shape 2가지 구성 하기 1,2번>
 (time_steps, feature)
 1. (784, 1) // [(392, 2), (186, 4)] :는 가능은 하다 대신에 값이 잘 안나오면 버리자 
 2. (28, 28) != (51, 5)
 : 쉐이프만 바꾸는 것은 데이터를 조작하지 않아
 : * 전체를 곱한 숫자만 같으면 된다 (리쉐이프 검증할 때 개념처럼) '''

''' 튜닝 값 (0.95 이상)
 <loss : categorical 사용 시>
 loss : 0.2606225961089134
 acc : 0.920199990272522
 '''

import numpy as np

# Datasets 불러오기
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0])                   # 0 ~ 255까지의 숫자가 적혀짐 (color에 대한 수치)
print('y_train : ', y_train[0])     # 5

print(x_train.shape)                # (60000, 28, 28)
print(x_test.shape)                 # (10000, 28, 28)
print(y_train.shape)                # (60000,)
print(y_test.shape)                 # (10000,)


# 데이터 전처리 1. OneHotEncoding
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)                # (60000, 10)

# 데이터 전처리 2. 리쉐이프 & 정규화
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# 2. 모델
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

model = Sequential()

model.add(LSTM(10, activation='relu', input_shape=(28,28)))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=50, mode='auto')

# 3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=128, verbose=2, validation_split=0.2,
          callbacks=[es])


# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=128)
print("loss :", loss)
print("acc :", acc)

''' SHAPE 설명
 (A,B,C,D,E,F,G,H,I,J)    # (10,)     A
                                      B
 A B C D E                            C
 F G H I T                # (2,5)     D
                                      E    # (10,1)
 AB                                   F
 CD                                   G
 EF                       # (5,2)     H
 GH                                   I
 IJ                                   J

 A B C D        # (2,4)
 E F G H

 [A],[B]
 [C],[D]        # (4,2,1)
 [E],[F]
 [G],[H] '''