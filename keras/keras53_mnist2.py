# 200527 0930~1000

''' 튜닝 값
 loss : 0.08200428349219147
 acc : 0.9776999950408936
 rmsprop '''

import numpy as np
import matplotlib.pyplot as plt

# Datasets 불러오기
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train[0])                     # 5의 숫자표현 이미지
print('y_train : ', y_train[0])         # 5
 # 두개의 값은 쌍일 것 F5 실행 확인
 # 각 컬럼마다 0~255의 숫자가 포함되어 있다. 0:하얀색, 255:가장 진한 검정색

print(x_train.shape)                # (60000, 28, 28)
print(x_test.shape)                 # (10000, 28, 28)
print(y_train.shape)                # (60000,)
print(y_test.shape)                 # (10000,)

print(x_train[0].shape)             # (28, 28) 이미지 사이즈
# plt.imshow(x_train[0], 'gray')
 # plt.imshow(x_train[0]) # 색을 제거 한 것
# plt.show()
 # (28, 28) 짜리가 6만장 // 가로 28픽셀 세로 28픽셀짜리 데이터

# mnist는 총 7만개의 28x28의 데이타 (0~255의 x값) = 현황 체크
''' 다중 분류 모델로 하려면 onehot 인코딩
 이진 분류 모델에서는 OneHot 인코딩은 필요없다. 0,1 떨어지니까
 (y값) 시작이 0~9까지 (카테고리컬이 편하겠네)
 0~9까지 손글씨 이미지 데이터가 7만장 모아논 데이터 셋
 그걸 수치로 바꾼 것이 x_train[0]는 '아 5구나!'
 0~9까지 10개니까 0~9사이에 들어가야지 0.99 안돼 0.1 안돼 1되지
 어떤 숫자가 나오건 0~9사이가 되어야하고 그래서 onehot인코딩 사용해야한다
 (수치 나오기 전에 출력을 해서 그 중에서 가장 큰 숫자 쪽으로 결정되는)
 Onehot인코딩도 전처리의 방식 '''
''' OnehotEncording 심층 설명
 지구인, 외계인, 강아지, 고양이, 악어, 치킨, 쥐, 드래곤 (8EA 데이터)
    0      1      2       3     4     5    6    7
 x.shape = (1만장, 32, 32, 1) : 이 안에는 픽셀이 255개인 데이터가 포함되어있다. 0~255까지
 y.shape = (1만장, ) : 스칼라가 1만장 짜리고 벡터가 1개 -> 현 상황에서 output_dim = 1
 y.shape.인코딩 = (1만장, 8) -> 최종 output = 8
 활성화 함수는 : softmax를 사용하게 된다. (아래가 그 원리)
 0.2 0.4 0.01 ... 0.003 = 다 +하면 1
  0   1    0  ...    0  = 이것이 외계인 이미지가 되겠다. '''


# 데이터 전처리 1. OneHotEncording
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)        # (60000, 10)
print(y_train)

# minmax가 왜 낫지? (질문) max 값을 모를 때는 민맥스
# 6만개 중에서 한땀한땀 찾을 수 없으니?

# 전처리 255. 방법2
 # x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
 # x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.
 # 이 방식도 가능하다.

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255
 # 민맥스 스칼라를 사용하지 않아도 저렇게 해서 같은 효과를 도출가능하다.
 # 실수 형태가 되기 때문에, 에러뜬다. 즉, 플롯 형태로 변환을 해줘야한다. (지금은 int형이기 때문)
 # x-최소/최대-최소
 # 리쉐이프 하고 타입 을 바꾸고 전처리까지 한 것
 # cnn 모델 인풋 쉐이프가 행무시하구 (가,세,채널) cnn 모델에 넣기 위해 리쉐이프
 # 안에 들어간 형태가 (x값) 0~255 astype로 실수형태로 변환 => 0~255. 이 되었다.
 # /255 는 정규화 민맥스와 거의 동일한 효과 (약간의 차이는 있음)
 # x값은 전처리되었다.

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(28,28,1)))       # (9, 9, 10)
model.add(Conv2D(20, (2,2), padding='same'))                               # (7, 7, 7)
model.add(Conv2D(30, (2,2), padding='same'))               # (7, 7, 5)
model.add(Conv2D(40, (2,2), padding='same'))               # (7, 7, 5)
model.add(Conv2D(30, (2,2), padding='same'))               # (7, 7, 5)
model.add(Conv2D(20, (2,2), padding='same'))                               # (6, 6, 5)
model.add(MaxPooling2D(pool_size=2))                      # (3, 3, 5) 2D 이기 때문에, 2x2로 그래서 풀_사이즈=2
model.add(Flatten())                                      # 45 = (3 * 3 * 5)
 # conv 레이어의 끝은 항상 Flatten을 넣어야 한다.
model.add(Dense(10, activation='softmax'))

model.summary()


# 3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
 # 2진 분류 : binary_crossentropy / 다중 분류 : categorical_crossentropy
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.25)


# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss :", loss)
print("acc :", acc)

# # x_pred = np.array([1,2,3])
# y_pred = model.predict(x_pred)                     # 'softmax'가 적용되지 않은 모습으로 나옴
# # y_pred = np.argmax(y_pred, axis=1)+1
# print(y_pred)
# print(y_pred.shape)                                # (3, 5)