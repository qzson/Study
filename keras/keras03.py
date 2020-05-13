# 1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10]) # ctrl + c 문단 복사 / shift + del 문단 삭제 / ctrl + 슬래시 문단 주석처리

# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense # DNN구조에 가장 베이스가 되는 구조 dense레이어.
model = Sequential() # 시퀀스를 앞으로 모델로 명시 하겠다.

model.add(Dense(5, input_dim = 1)) # activation 은 디폴트가 있다.
model.add(Dense(3))
# model.add(Dense(1000000))
# model.add(Dense(1000000)) # 여기서 막힘, GPU에서 또 할건데 막히는지 안막히는지. gpu는 메모리가 많고 cuda코어 많은 것. 텐서코어 많은 것이 좋은 것.
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
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
loss, acc = model.evaluate(x, y) # 평가를 loss와 acc에 받겠다. / 
print("loss :", loss) # loss가 1이 넘어도 좋은 것이 있고 0.1이어도 안좋은 것일 수 있다.
print("acc :", acc)