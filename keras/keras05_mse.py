# 1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10]) # ctrl + c 문단 복사 / shift + del 문단 삭제 / ctrl + 슬래시 문단 주석처리
x_pred = np.array([11, 12, 13]) # 모델이 좋은지 확인하기 위해 x_pred를 넣고 y값은 예측할 것이다. y2가 11, 12, 13 나올 것 같네

# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense # DNN구조에 가장 베이스가 되는 구조 dense레이어.
model = Sequential() # 시퀀스를 앞으로 모델로 명시 하겠다.

model.add(Dense(5, input_dim = 1)) # activation 은 디폴트가 있다.
model.add(Dense(3))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(1))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x, y, epochs=30, batch_size=1)

# 4. 평가, 예측
loss, mse = model.evaluate(x, y) # 평가를 loss와 acc에 받겠다. / 
print("loss :", loss) # loss가 1이 넘어도 좋은 것이 있고 0.1이어도 안좋은 것일 수 있다.
print("mse :", mse)

y_pred = model.predict(x_pred) # model.pred는 y_pred를 반환하게 되어있다.
print("y_predic :", y_pred) # 10.999가 11과 같은 걸까? 좋은 건가? acc는 1.0인데? 뭐가 문제가 일까? 인공지능 방식에는 두가지 방식이 있다. 회귀방식(리뉴어 리그래서) 분류방식이 두가지가 있다.

''' #

loss는 좋아지고 acc도 좋아진다.
훈련상으로는 좋은 걸 느낄 수 있다. (소스상으로는 오류 소스지만,)
수치를 봐라. 떨어지고 있다. 올라가고 있다.를 보면 알 수 있다.
하지만, 또 잘못된 점이 있다.
model.fit 으로 이미 훈련을 시켰다. 1~10까지. 그리고 model.evaluate에 동일한 값을 또 넣었다.
시험전날 모의고사 보고 시험날 똑같이 나온 것과 같은 것
데이터셋을 받을 때, 훈련 데이터와 평가용 데이터를 잘라내야 해
훈련 데이터와 평가 데이터는 같은 값을 쓰면 안된다

'''