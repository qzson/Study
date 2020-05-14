# 1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16, 17, 18])

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
model.add(Dense(1))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=200, batch_size=1)

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test)
print("loss :", loss)
print("mse :", mse)

# y_pred = model.predict(x_pred)
# print("y_predic :", y_pred)

y_predict = model.predict(x_test)
print(y_predict)

from sklearn.metrics import mean_squared_error # mse 를 sklearn.metrics 에서 땡겨오겠다.
def RMSE(y_test, y_predict): # RMSE로 정의하겠다. 원래 y값 테스트와 y프레딕트를 땡겨올거고
    return np.sqrt(mean_squared_error(y_test, y_predict)) # np.sqrt(루트)를 리턴해줄거다
print("RMSE : ", RMSE(y_test, y_predict))

# 판단기준이 2개가 되었다. MSE, RMSE 이두가지. // RMSE는 해커톤이나 케글에서 가장 많이 쓰이는 함수
# def 함수 다음() 부분이 입력값 return 이후는 출력값 그래서 함수 RMSE를 기입해주고 입력값에 인자들을 넣어주면
# 재사용이 가능하다. 즉 함수화 하여 조금씩 갖고 있어라.
# 모델 구성 부분도 함수로 간결화 할 수 있다.
# 지표 mse를 썼지만, RMSE도 쓴다는 
# 회귀모델에서 중요한 지표가 하나 있다. R2
# 내가 만든 모델에 R2를 적용 했을 때 높으면 높을수록 좋다. 최대값은 1 / mse와 rmse는 반대
# 내가 만든 모델에 r2와 mse나 rmse 중 하나를 사용. 두 개를 비교하며 좋은 모델링이 되었는지 판단
# 이 역시 함수를 만들어 재사용 가능

'''

※ MSE  : Mean Squared Error
: [sigma(실제값 예측값)^2 /n]

※ RMSE : Root Mean Squared Error, 평균 제곱근 오차
: root[sigma(실제값 - 예측값)^2 /n]

* mse에 비해 에러값의 크기가 실제 값에 비례한다. (제곱된 것에 루트를 씌웠기 때문)
* mse는 실제값과 예측값의 차이를 제곱해 평균한 것

RMSE 수는 낮을수록 좋다.
사이킷런 : 텐서플로 케라스가 나오기전 머신러닝 분야에서 킹왕짱하던 놈이다
사이킷런도 케라스처럼 api 이며 거기서 RMSE를 가져다 쓸거다.
함수의 목적 : 재사용을 하기 위해서.

'''