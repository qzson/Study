# 200514 11:15~
# keras09_test에서는 일부로 나쁜 모델을 만들어 실험을 했으니, 이번 keras10에서는 좋은 모델을 다시 만들어보자

# 1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
# x_pred = np.array([16, 17, 18])
x_val = np.array([101, 102, 103, 104, 105])
y_val = np.array([101, 102, 103, 104, 105])
'''
# 총 데이터 수는 20개 / w값은 몇 이야? 1일 것 같지? 아님. 통밥으로 얘기해줄 수 있는 사람 없을 걸 (val 때문에)
# 현 데이터 값은 [1~15,101~105](=20개) 인 것. 이걸 그래프로 표현하면 w = 1로 표현되다가 더 급격하게 올라간다. 붙어있는 데이터
'''

# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 1))
model.add(Dense(25))
model.add(Dense(50))
model.add(Dense(250))
model.add(Dense(500))
model.add(Dense(250))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(5))
model.add(Dense(1)) # RMSE : 8.844011779304222e-06 // R2 : 0.9999999999608917

# model.add(Dense(5, input_dim = 1))
# model.add(Dense(50))
# model.add(Dense(250))
# model.add(Dense(500))
# model.add(Dense(1000))
# model.add(Dense(500))
# model.add(Dense(250))
# model.add(Dense(50))
# model.add(Dense(5))
# model.add(Dense(1)) - 이거는 왜 값이 위에 것 보다 안좋을까? 노드 개수도 많은데..

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val))
'''# val을 fit에 적용.'''

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test)
print("loss :", loss)
print("mse :", mse)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# RMSE 수치는 0에 가깝게 R2는 1에 가깝게 해야 좋은 모델
# val 을 사용함으로써 더 좋은 모델이 만들어진다.
# 터미널창을 보면 val_loss, val_mse 등이 보인다. val을 넣었기 때문에 적용이 되기 시작했다는 것 확인 가능
'''
보면 loss와 val_loss 틀려. 왜냐 할때 마다 검증하잖아,
통상적으로 훈련(공부)할 때보다 평가(시험)했을 때 결과치(성적)이 떨어지지?
그래서 loss가 상대적으로 val_loss 구간이 더 높아
그리고 RMSE나 R2는 val구간이 좀 더 낮다.? - 이거 맞나여
'''
# 지금은 데이터의 갯수가 너무 적어. 하지만 프로젝트시 데이터가 엄청 많기 때문에 지금 연습해보고 나중에 고생을 줄이자.

# 데이터 분포그래프
'''
105|-------------------------------.
104|                           /   |
103|                       /       |
102|                  /            |
101|--------------.                |
.  |             /|                |
.  |             /|                |
.  |            / |                |
15 |-----------.  |                |
   |        /  |  |                |
   |      /    |  |                |
   |   /       |  |                |
   | /         |  |                |
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    1 2 3 ...  15 101 102 103 104 105
'''
# w:     1,      A,        1         => 붙어있는 데이터이기 때문에 가중치가 다르다.
