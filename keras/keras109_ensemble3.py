# 20-07-02_27
# 한 개의 모델에서 분류와 회귀를 동시에 나오게 할 수 있을까?
# 컴파일에서 한가지 추가 = loss_weights


### 1. 데이터
import numpy as np

x1_train = np.array([1,2,3,4,5,6,7,8,9,10])
x2_train = np.array([1,2,3,4,5,6,7,8,9,10])
y1_train = np.array([1,2,3,4,5,6,7,8,9,10])
y2_train = np.array([1,0,1,0,1,0,1,0,1,0])


### 2. 모델
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate

input1 = Input(shape=(1,))
x1 = Dense(128)(input1)
x1 = Dense(64)(x1)
x1 = Dense(32)(x1)

input2 = Input(shape=(1,))
x2 = Dense(128)(input1)
x2 = Dense(64)(x2)
x2 = Dense(32)(x2)

merge = concatenate([x1, x2])

x3 = Dense(32)(merge)
output1 = Dense(1)(x3)

x4 = Dense(32)(merge)
x4 = Dense(16)(x4)
output2 = Dense(1, activation='sigmoid')(x4)

model = Model(inputs = [input1, input2], outputs=[output1, output2])
model.summary()


### 3. 실행, 훈련
model.compile(loss = ['mse', 'binary_crossentropy'], optimizer='adam', metrics=['mse', 'acc'],
              loss_weights=[0.1, 0.9])
              # loss에 대한 가중치 비율을 두 번째 모델에 90%를 주겠다.

model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=100, batch_size=1, verbose=2)


### 4. 평가, 예측
loss = model.evaluate([x1_train, x2_train], [y1_train, y2_train])
print('loss :', loss)
# 그럼에도 결과는 역시 또또 좋지 않았다. (회귀 모델은 잘 맞았는데, 분류 모델은 잘 안 맞았다.)
# > 그냥 모델 2개 써라. ㅋ_ㅋ
# >> 앙상블 자체가(여러분들이 만든 모델들이) 꼭 좋은 것은 아니다.
# >>> 전체 loss가 단순 더하기의 반환이었다는 것 잊지 말고
# >>> 2,3번째 loss가 각 모델별 loss 합이 였는데, 이것은 loss_wegiht 때문에 다르게 합해줘야한다.
# >>> 2에 * 0.1 + 3에 * 0.9 = 전체 loss 


x1_pred = np.array([11, 12, 13, 14])
x2_pred = np.array([11, 12, 13, 14])

y_pred = model.predict([x1_pred, x2_pred])
print(y_pred)
