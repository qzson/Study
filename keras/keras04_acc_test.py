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

y_pred = model.predict(x_pred) # model.pred는 y_pred를 반환하게 되어있다.
print("y_predic :", y_pred) # 10.999가 11과 같은 걸까? 좋은 건가? acc는 1.0인데? 뭐가 문제가 일까? 인공지능 방식에는 두가지 방식이 있다. 회귀방식(리뉴어 리그래서) 분류방식이 두가지가 있다.

''' #

1. 회귀 (선형방식)
x가 1일 때, y가 1 ... 선을 쭉 그어, 근데 아까 훈련 했더니 x-11에 y-11.005 식으로 나왔다.
즉, w가 0.9999 or 1.00001 이다. 11,12,13 했는데 같은 값이 아니니 acc는 0이다.
그치만, 머신은 맞데, 사실 회귀는 수치로 답을 해준다. 수치가 틀리면 틀리게 나와야한다. 그렇기에 오류다.
loss를 봅시다. mse는 (구글 키고 mse를 찾아.)
mse는 회귀지표인데, acc는 분류지표이다. 그래서 안 맞는 거다.
분류라는 것은 y값에대한 분류에 대한 범위를 정해놔야함
즉, acc로 분류를 쓰려면 y값이 고정이 되어야한다.
mse를 했기 때문에 고정이 아니라 자유롭게 나온다.
메트릭스 소스를 바꿔야한다? 단지 이 소스는 오류를 확인하기 위한 소스

'''