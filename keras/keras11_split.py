# 200514 12:20~

# 1. 데이터
import numpy as np
x = np.array(range(1,101))
y = np.array(range(101,201))
'''
range 라는 함수 : (비전공자의 시각) 시작점은 1 특성상 뒤쪽은 -1 값 까지
range(a, b) : a에서 (b-1)까지 생성 // a값을 설정 안할 시 0
만들어진 후 부터 0부터 시작되는 순서 부여된다.
'''
x_train = x[:60] # [:60] => (pirnt값은 1~60) 원리: 처음(0)부터 59까지 (뒷 숫자 -1 가 들어간 다는 것 참고 아래 서술)
x_val = x[60:80] # 20개 (print값은 61~80)
x_test = x[80:] # 81부터 이건 끝까지라는 뜻. (이걸 통해서 train:val:test = 6:2:2임을 알 수 있다.)
''' 
[] 값 사이 기입 시 주의 : 61부터 시작인지, 60부터 시작인지.
그렇지만, 사실 1~2개 정도 틀려도 모델은 돌아간다 (100만개 중에서 1~2개 틀려도 상관없다는 모순?)
python 시퀀스 자료형 슬라이스 참조

* 실제 : 1 2 3 4 5 6 7 8 ~ 58 59 60 61 62 ~ 78 79 80 81 82 ~ 98 99 100
* 배열 : 0 1 2 3 4 5 6 7 ~ 57 58 59 60 61 ~ 77 78 79 80 81 ~ 97 98  99

# x[:60] = 1 ~ 60
#   배열 : 0 ~ 59까지 출력
# 배열의 순서로 따짐 : 0부터 (60 - 1)번 째 까지 

# x[60:80] = 61 ~ 80
#     배열 : 60 ~ 79 (80 - 1)            

# x[80:] = 81 ~ 100
#   배열 : 80 ~ 99
'''

y_train = x[:60]
y_val = x[60:80]
y_test = x[80:]

print(x_train)
print(x_val)
print(x_test)
''' range 값이 1~100까지니까. (x[:60] = 인덱스 0 ~ 인덱스 59 > 60개 즉 레인지 범위에서 앞(1)에서부터 60개 빠진다라는 뜻)
    x[60:80] = 60번째 인덱스 부터 79까지 > 20개 즉, 20개 빠진 것
'''


# 2. 모델구성 (와꾸 맞아서 모델 그냥 사용 = 전이학습-transfer learning)
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 1)) # 1~60까지의 한 덩어리가 input에 들어간다고 생각해
model.add(Dense(25))
model.add(Dense(50))
model.add(Dense(250))
model.add(Dense(500))
model.add(Dense(250))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(5))
model.add(Dense(1)) # 나가는 와꾸도 맞아

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=1000, batch_size=1,
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