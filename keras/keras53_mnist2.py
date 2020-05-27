# 200527 0930~1000

''' 튜닝 값
 loss : 0.08200428349219147
 acc : 0.9776999950408936
 rmsprop
 
 loss : 0.17600808977459256
 acc : 0.9739000201225281
 adam 사용 '''

import numpy as np
import matplotlib.pyplot as plt

# Datasets 불러오기
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0])
print('y_train : ', y_train[0])

print(x_train.shape)                # (60000, 28, 28)
print(x_test.shape)                 # (10000, 28, 28)
print(y_train.shape)                # (60000,)
print(y_test.shape)                 # (10000,)

# mnist는 총 7만개의 28x28의 데이타 (0~255의 x값) = 현황 체크


''' 다중 분류 모델로 하려면 OneHotEncoding
 이진 분류 모델에서는 OneHot 인코딩은 필요없다. (0,1 떨어지니까)

 불러온 데이터 기준 // y값 = 0~9 까지 (카테고리컬이 편할 것 예상) 0~9 까지 손 글씨 이미지 데이터가 7만장 모아논 데이터 셋
 0~9 까지 10개 이므로, 0~9 사이에 들어가야 한다 // 0.99 안돼 / 0.1 안돼 / 1 가능
 어떤 숫자가 나오건 0~9 사이가 되어야하고 그래서 OneHotEncoding 을 사용해야 한다
 (수치 나오기 전에 출력을 해서 그 중에서 가장 큰 숫자 쪽으로 결정되는)
 OneHotEncoding 도 전처리의 방식 '''

''' OnehotEncoding 심층 설명
 
 지구인, 외계인, 강아지, 고양이, 악어, 치킨, 쥐, 드래곤 (8EA 이미지 데이터)
    0      1      2       3     4     5    6    7
 
 x.shape = (1만장, 32, 32, 1) : 이 안에는 픽셀이 255개인 데이터가 포함되어있다. 0~255까지
 y.shape = (1만장, ) : 스칼라가 1만장 짜리고 벡터가 1개 -> 현 상황에서 output_dim = 1
 y.shape.인코딩 = (1만장, 8) -> 최종 output = 8
 활성화 함수는 : softmax를 사용하게 된다. (아래가 그 원리)
 0.2 0.4 0.01 ... 0.003 = 다 + 하면 1
  0   1    0  ...    0  = 이것이 외계인 이미지(1번 인덱스)가 되겠다. '''


# 데이터 전처리 1. OneHotEncoding
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)        # (60000, 10)

# 데이터 전처리 2. 정규화 (MinMaxScalar)
 # minmax가 왜 낫지? (질문) max 값을 모를 때는 민맥스
 # 6만개 중에서 한땀 한땀 찾을 수 없으니?

 # x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
 # x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.
 # 이 방식도 가능하다.

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255
 #             CNN 사용을 위한 4차원 설정     # 타입 변환     # (x - min) / (max - min) : max =255, min = 0 
 
 # 민맥스 스칼라를 사용하지 않아도 저렇게 해서 같은 효과를 도출가능하다.
 # 실수 형태가 되기 때문에, 에러뜬다. 즉, 플롯 형태로 변환을 해줘야한다. (지금은 int형이기 때문)
 # 리쉐이프 하고 타입 을 바꾸고 전처리까지 한 것
 # cnn 모델 인풋 쉐이프가 행 무시하고 (가,세,채널) cnn 모델에 넣기 위해 리쉐이프
 # 안에 들어간 형태가 (x값) 0~255 / astype로 실수 형태로 변환 => 0~255. 이 되었다.
 # /255 는 정규화 민맥스와 거의 동일한 효과 (약간의 차이는 있음)
 # x 값은 전처리 되었다.

# 2. 모델
# 0 ~ 9까지 씌여진 크기가 (28*28)인 손글씨 60000장을 0 ~ 9로 분류하겠다. ( CNN + 다중 분류)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(10, (2,2), padding='same', input_shape=(28,28,1)))
model.add(Conv2D(20, (2,2), padding='same'))
model.add(Conv2D(30, (2,2), padding='same'))
model.add(Conv2D(40, (2,2), padding='same'))
model.add(Conv2D(30, (2,2), padding='same'))
model.add(Conv2D(20, (2,2), padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))            # 다중 분류

model.summary()


# 3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=30, batch_size=64, verbose=1, validation_split=0.2)


# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=64)
print("loss :", loss)
print("acc :", acc)


'''
# 자습 : x_test를 10행 가져와서 x_predict로 써보기
x_pred = x_test[:10]
print(x_pred.shape)                                       # (10, 28, 28, 1)
y_pred = y_test[:10]

y1_pred = np.argmax(y_test[:10], axis=1)                  # x_predict값에 매칭되는 실제 y_predict값
print('실제값: ',y1_pred)                                 # 실제값:  [7 2 1 0 4 1 4 9 5 9]

y2_pred = model.predict(x_pred)                           # x_predict값을 가지고 예측한 y_predict값
y2_pred = np.argmax(y2_pred, axis =1)
print('예측값: ', y2_pred)                                # 예측값:  [7 2 1 0 4 1 4 9 5 9]       

# acc:  0.98580002784729 '''