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
 # 컬럼은 2가지 밖에 없다. 아웃풋은 2가지 중 하나 0,1
 # 그래서 activation을 sigmoid라는 것을 써야한다.

print(x.shape)              # (10,)
print(y.shape)              # (10,)


# 2. 모델
model = Sequential()
model.add(Dense(256,activation = 'elu', input_dim = 1))
model.add(Dense(128,activation = 'elu'))
model.add(Dense(64,activation = 'elu'))
model.add(Dense(32,activation = 'elu'))
model.add(Dense(16,activation = 'elu'))
model.add(Dense(1,activation = 'sigmoid'))
""" 설명
 - 계산된 함수가 activation을 통해 다음 layer에 넘어간다.
 - 가장 마지막 output layer값이 가중치와 '활성화 함수'와 곱해져서 반환된다. 
 # sigmoid : 출력 값을 0과 1사이의 값으로 조정하여 반환한다. """


# 3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
 # acc : 0~1, 1과 0을 얼마나 잘 찾았는가 정확도를 찾는 것
 # 분류 모델에서는 평가 모델을 acc를 사용한다. 딱 떨어지는 것
model.fit(x, y, epochs=500, batch_size=32, verbose=2)
 
''' binary_crossentropy
 
 2진 분류에서는 binary_crossentropy를 사용하고 있다. (loss 값은 이것으로 치환) 외워라.
 매트릭스에 mse 넣어도 돌아는 가지만, 쓸 필요는 없다. 결과치가 어차피 0~1 사이일 것이다.
 다중 분류 모델에서는 또 틀려진다.
 
 실질적으로 가장 중요한 것은 acc : 0.5
 
 <과제 1. y_pred 값이 0과 1로 출력되게 나오게 조정>
 방법1. 한땀 수정하여 만들어본다.
 방법2. 남이 잘 만들어 놓은 것을 찾는다.
 방법3. 찾아내 '''


# 4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=32)
print("loss :", loss)
print("acc :", acc)

x_pred = np.array([1,2,3])
y_pred = model.predict(x_pred)
print(y_pred)
# sigmoid 함수를 거치지 않은 걸로 보여짐

y1_pred = np.where(y_pred >= 0.5, 1, 0)     
print('y_pred :', y1_pred)

""" where
 # np.where(조건, 조건에 맞을 때 값, 조건과 다를때 값)
 : 조건에 맞는 값을 특정 다른 값으로 변환하기 """