# 200514 14:20~ 15:00~ 데이터 스플릿

# 1. 데이터
import numpy as np 
x = np.array(range(1,101))
y = np.array(range(101,201))
'''
# ★<w = 1, bais = 100 (왜지??)> 집단 린치 당함. // y=wx+b 인데, 101=w1+b 는 ..? 당연히 b=100 '''

# train_test_split(sklearn)사용하여  train, test, validation 나누기

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    # x, y, random_state = 99, shuffle = True,
    x, y, shuffle = False,
    train_size = 0.6)

# x, y의 전체 데이터 받기, train_size를 전체 데이터의 60% 받겠다.

'''
# tts는 x,y를 받고 random_state : // test_size : 40% or train_size : 60% 로 설정 가능
# random 은 컴퓨터에서 랜덤 난수가 있다. 랜덤난수표 66번에 해당하는 것을 가져와서 섞는 것이다.
# random_state를 잡지 않고 하면 할 때마다 바뀐다.
# shuffle 은 과거에 일어나지 않은 부분들에 의해서 결과치가 변할 수 있기 때문에 전체 데이터를 섞어서 해줌으로써 조금 더 정확한 결과치를 만들어낸다.
# shuffle 의 defalut = True '''
# shuffle을 하는 이유? (구자님 git 부연설명)
# : train와 test data의 범위가 완전히 분리되어 있으면 test값(train범위 외의 구간)을 제대로 유추 못할 수 있다.
#   ( 한번도 Train에서 경험하지 못했기 때문에)
# : 그래서 train과 test data범위가 겹치는 것이 정확도를 올리는데 좋음
# shuffle 조건
# : x, y를 쌍으로 넣어야 함 -> x와 y가 매칭되어야 하기 때문에


x_val, x_test, y_val, y_test = train_test_split(
    # x_test, y_test, random_state = 99,
    x_test, y_test, shuffle = False,
    test_size = 0.5)

# x_test, y_test 데이터 받아서 그 데이터 50%를 test_size로 설정 

'''
# ★<x_test는 왜 기입?> // 위에서 x,y로 한번 전체 데이터 값에 대한 60% 할당을 진행했다.
# 그곳에서 x,y_test 값은 40%로 할당이 되었으므로 그 40% 중 0.5이니 20% 로 할당하겠다는 의미.'''


# x_train = x[:60]
# x_val = x[60:80]
# x_test = x[80:]

# y_train = x[:60]
# y_val = x[60:80]
# y_test = x[80:] 이거는 우리가 가내수공업 한 것 (keras11_split 참고)

print(x_train)
print(x_val)
print(x_test)

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
model.add(Dense(1))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val))

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
'''