from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train = np.array([1,2,3])
y_train = np.array([4,5,6])

model = Sequential()
model.add(Dense(5, input_dim = 1, activation='relu'))
model.add(Dense(3))
model.add(Dense(1, activation='relu'))

# 인풋 아웃풋 사이에 히든레이어 dim(디멘션 -차원)

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100)
# 원래 뱃치 사이즈에 1 값이 있었지만, 지워본다. 지워도 돌아가니 기본 값이 있다는 것. 기본 값을 찾아라.
# batch_size의 기본값은 32.
# 아래 evaluate는 무시하고 일단 진행 13일에 강의.
# batch_size의 기본값이 32인 이유를 유추해본다면, 일반적으로 우리가 데이터를 받아 돌릴때,
# 32라는 수치가 제일 적당하기 때문에 그럴 것이다 라고 예상.
# 그리고 이상적으로는 1로 잡고 하는 것이 가장 정확할 것 같지만 그렇지 않다.
# 과적합? 문제가 있을 수 있고 시간이 너무 오래걸린다.

loss, acc = model.evaluate(x_train, y_train, batch_size = 1)

print("loss: ", loss)
print("acc : ", acc)

output = model.predict(x_train)
print("결과물 : \n", output)

# 레이어, 에포치스를 수정을 통해 accuracy 값을 1에 가깝게 만드는 것이 중요.

# x 123 y 246 식으로..?
