# 200520

from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# import numpy as np
# x = np.array

# 1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])    # (4,3)
y = array([4,5,6,7])                            # (4, ) : 스칼라 4, 벡터 1 // (4, ) != (4,1)
# y2 = array([[4,5,6,7]])                         # (1,4)
# y3 = array([[4],[5],[6],[7]])                   # (4,1)

print("x.shape:", x.shape)                      # (4,3)
print("y.shape:", y.shape)                      # (4, )

# 계산법 : 가장 큰 대괄호 걷어내고, 가장 큰 최소 값, 그 다음 괄호 묶이는 값
#         오른쪽 에서 왼쪽으로 기입

# x = x.reshape(4, 3, 1)                      # (4,3,1)로 모양을 바꿔준다.
x = x.reshape(x.shape[0], x.shape[1], 1)      # (정석) 이렇게 했을 때, 쉐이프에 맞춰서 바꿔줄 필요가 없다
print(x.shape)                                
# reshape 검사시 값을 다 곱해서 처음 행렬 값과 같으면 문제 없다.
# 4행 3열짜리를 1개씩 작업을 하겠다. (4,3,1) - LSTM에 있는 데이터를 몇개씩


# 2. 모델구성
model = Sequential()
model.add(LSTM(30, activation='relu', input_shape=(3,1)))
# 행의 개수는 중요하지 않다(데이터의 수). 컬럼이 데이터의 종류니까
# 즉, input_shape는 컬럼과 몇개씩 잘라서 작업을 할 건지 묻는 것.
# 아래 부터는 Dense 모델
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1)) # 1개 예측 y = [4,5,6,7]

model.summary()


# 3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=100, batch_size=1)

x_input = array([5,6,7])            # 5,6,7 을 통해서 8을 나오게 만들어보고 싶다.
x_input = x_input.reshape(1,3,1)    # (3, )
# 1,3,1 중에 3,1은 인풋 쉐이프 3,1 맞춰주고 앞의 1은 와꾸 맞추기 위해서
# 와꾸가 안맞음--->(1,3,1)로 변환 (행, 열, 몇개로 쪼갤건지)

print(x_input)

# 4. 예측
yhat = model.predict(x_input)
print(yhat)
# 정확하게 예측이 안된다. LSTM너무 적어서 , 수정할 수 있는 부분 수정


'''
X     |     Y
123         4
234         5
345         6
456         7
(4,3) |  (4, ) : 스칼라가 4라는 뜻

대괄호 걷어내고
스칼라 벡터 행렬 텐서의 개념 잡기
스칼라:
y는 4한개가 스칼라, 5한개가 스칼라...
x도 1,2,3 각각 스칼라.
벡터:
y는 스칼라(4) 나열 1개

1. [[1,2,3],[1,2,3]]                (2,3) 2차원텐서
2. [[[1,2],[4,3]],[[4,5],[5,6]]]    (2,2,2) 3차원텐서
3. [[[1],[2],[3]],[[4],[5],[6]]]    (2,3,1)
4. [[[1,2,3,4]]]                    (1,1,4)
5. [[[[1],[2]]],[[[3],[4]]]]        (2,1,2,1)

'''