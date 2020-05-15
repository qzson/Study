# 200515, 0900 ~ / 

'''
아웃풋을 많이 늘릴 수 있다?
'''

# 1. 데이터
import numpy as np
x = np.transpose([range(1,101), range(311, 411), range(100)])
y = np.transpose([range(101,201), range(711,811), range(100)])
# 100개 짜리 x=3덩이

# print(x.shape)

# print(x)
# 이 상태로 출력 안된다. (문법이 틀렸기 때문에) : x = np.array(range(1,101), range(311, 411), range(100)) (x)

# 파이썬에 list라는 것이 있다. 내가 할 수 있는 것에 대해서 모았다는 의미.
# 머신이 하나의 리스트이다. 라고 인식하게 해줘야 한다. (대괄호 씌워야한다.)
# print(y)
# x,y 을 각 덩어리마다 비교해보면 w 값이 일정하니까, 정제된 깔끔한 데이터다.
# 데이터에 이상이 없다는 것을 확인 가능

'''
* (3, 100) 의 설명

행(row), 열(column 컬럼)
★ 외우기 : '열우선, 행무시' ☆ 머리속에 기억 ☆

예)
                    X                     |     Y
    <날씨(온도)>    용돈(쓴돈)  삼성주식    |     SK
1/1     15            100       2만             1천
1/2     16            1000      1.5             2천
 *
 *
 *
12/31   -30          10000      10만      |     5만
=
<(365일)365행, 1열> 의 데이터들.
각 컬럼 : 날씨, 용돈, 삼성주식
365데이터에 1컬럼.
X = (365, 3)
Y = (365, 1) 임.
------------

이전 데이터 중에,
(10, 1) train
(5, 1)  test
(3, 1)  pred
잘 돌아갔지? 데이터 전체 갯수는 상관없다. - 행을 무시했다는 것. 중요한 것은 열이 한개씩이었다.

(365, 3)
여기서 컬럼은 3 즉, = 'input.dim = 3' (신경망 입력값이 3이 되는 것이다)
여기서 컬럼 추가 시, 문제 생길 수 있다. (쉐이프에 - 적어놓고 무슨 말인지 잘 모르겠다 - 알려주실 분?)

(3, 100)
1   ~ 100
311 ~ 410
0   ~  99
가로로 표시되기 때문에 세로로 바꿔줘야 한다. (배열 전치가 필요함)
따라서, 100행 3열로 바꿔야 한다. 우리의 의도대로. (3, 100) -> (100, 3)
그 함수는?
1. transpose
2. reshape
3. swapaxes
가 있다. (자주 쓰이는건 transpose 인가? 이후 수업에서 transpose 를 사용)

'''

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = False,
    train_size = 0.8)

print(x_train)
# print(x_val)
print(x_test)


# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 3))
model.add(Dense(25))
model.add(Dense(50))
model.add(Dense(250))
model.add(Dense(500))
model.add(Dense(250))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(5))
model.add(Dense(3))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.25)

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