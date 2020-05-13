# 200513 9:41 강의 설명

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

'''
keras.models ~ = keras>models>sequential 케라스 안에 모델스 안에 시퀀셜을 쓰겠다.
클래스안에 클래스안에 클래스
이 Sequential이라는 것을 모델을 명시해줘야한다. (model = Sequential)
그 시퀀셜 안에 add로 추가한다.
.은 상단 것에서 가져와 땡겨쓴다.

'''

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([101,102,103,104,105,106,107,108,109,110])

model = Sequential()
model.add(Dense(5, input_dim = 1, activation='relu'))
model.add(Dense(3))
model.add(Dense(1, activation='relu'))

model.summary()

# -------------여기까지 신경망 모델링---------------

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# 우리 모델을 컴파일 시킨다. 시퀀셜안에 모델이 있겠지 시퀀셜이라는 걸 모델이라는 변수로 잡았다.
# loss 손실. 낮은게 좋다. 손실율을 적게 하는 건 mse로 하겠다. optimizer최적화는 adam으로 하겠다. metrics 그 부분을 accuracy로 명시

model.fit(x_train, y_train, epochs=100, batch_size = 1,
validation_data = (x_train, y_train))
# 모델 핏 훈련시키는 것. 나는 x트레인과 y트레인으로 훈련을 시킨다. 에포 100, 100번 훈련시킨다. 
loss, acc = model.evaluate(x_test, y_test, batch_size = 1)

print("loss : ", loss)
print("acc : ", acc)

output = model.predict(x_test)
print("결과물 : \n", output)

# 애큐러시가 0이 나올 수 있다. 데이터가 많지 않기 때문에.?

# x = 10개 batch_size 1에 epochs 1 이면 10번 작업