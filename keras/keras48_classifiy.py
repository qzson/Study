# 200526 0900~ // 1000~ 2진분류
# train test 분리 X
# 모델을 짜라
''' 과제1
 <과제 1. y_pred 값이 0과 1로 출력되게 나오게 조정>
 방법1. 한땀 수정하여 만들어본다.
 방법2. 남이 잘 만들어 놓은 것을 찾는다.
 방법3. 찾아내 '''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense


# 1. 데이터
x = np.array(range(1, 11))
y = np.array([1,0,1,0,1,0,1,0,1,0])
# 컬럼은 2가지 밖에 없다. 아웃풋은 2가지중 하나 0,1
# 그래서 activation을 sigmoid라는 것을 써야한다.
# loss는 mse 사용했었었는데, 2진 분류가 되면, 거기에 들어가는 
# sigmoid 방식 확인하고 // 바이너리? 방식을 확인해라.

print(x.shape)
print(y.shape)


# 2. 모델
model = Sequential()

model.add(Dense(5, input_dim = 1))
model.add(Dense(25, activation='relu'))     # relu 사용 시 85점정도는 먹힌다? (좋은 성능을 가진 relu)
model.add(Dense(50, activation='relu'))
model.add(Dense(20))                        # activation의 디폴트 값이 있다.
model.add(Dense(25, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))   # 마지막 아웃풋은 sigmoid


# 3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# acc : 0~1, 1과 0을 얼마나 잘 찾았는가 정확도를 찾는 것 / 분류모델에서는 평가모델을 acc를 사용한다. 딱 떨어지는 것
model.fit(x, y, epochs=100, batch_size=1, verbose=2)
# binary_crossentropy 찾아보자

''' binary_crossentropy
 
 2진 분류에서는 binary_crossentropy를 사용하고 있다. (loss 값은 이것으로 치환) 외워라.
 매트릭스에 mse 넣어도 돌아는 가지만, 쓸 필요는 없다.
 다중 분류 모델에서는 또 틀려진다.
 결과치가 어차피 0~1 사이일 것이다. 그런데 sigmoid를 거치지 않았다.
 왜 일까?
 
 실질적으로 가장 중요한 것은 acc : 0.5
 
 <과제 1. y_pred 값이 0과 1로 출력되게 나오게 조정>
 방법1. 한땀 수정하여 만들어본다.
 방법2. 남이 잘 만들어 놓은 것을 찾는다.
 방법3. 찾아내 '''


# 4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print("loss :", loss)
print("acc :", acc)

x_pred = np.array([1,2,3])
y_pred = model.predict(x_pred)
print(y_pred)