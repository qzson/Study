# 200519 9000 ~ 1015 ~ 설명
# 데이터 셋 개수[x = 1, y = 2]
# concatenate 사용은 못한다.
# < 하이퍼 파라미터 튜닝 : RMSE = 0.0435, R2 = 0.999997 >

'''
     (100, 2)
        ㅁ
   ㅁ        ㅁ
(100, 2), (100, 2)
'''

# 1. 데이터
import numpy as np
x1 = np.array([range(1,101), range(301,401)])

y1 = np.array([range(711,811), range(611, 711)])
y2 = np.array([range(101,201), range(411, 511)])

x1 = np.transpose(x1)
y1 = np.transpose(y1)
y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, y1, y2, random_state = 66, shuffle = True,
    # x1, y1, y2, shuffle = False,
    train_size = 0.8)

# shape 를 사용하는 버릇을 들이자.
# print(x1_train.shape) # (80, 2)
# print(y1_train.shape) # (20, 2)
# 체크하고 정상임을 확인


# 2. 모델구성
from keras.models import Sequential, Model
# 시퀀셜 지워도 된다. 안쓰니까
from keras.layers import Dense, Input

######## 모델 1 ########
input1 = Input(shape=(2,))
dense1_1 = Dense(4, activation='relu', name='dense1_1')(input1)
dense1_1 = Dense(6, activation='relu', name='dense1_2')(dense1_1)
dense1_1 = Dense(8, activation='relu', name='dense1_3')(dense1_1)
dense1_2 = Dense(10, activation='relu', name='dense1_4')(dense1_1)

# ######## 모델 2 (사용 X) ########
# input2 = Input(shape=(3,))
# dense2_1 = Dense(8, activation='relu')(input2)
# dense2_2 = Dense(4, activation='relu')(dense2_1)

# ######## 모델 병합 (사용 X - middle 레이어도 구성 필요 X) ########
# from keras.layers.merge import concatenate
# merge1 = concatenate(dense1_2)

# middle1 = Dense(30)(merge1)
# middle1 = Dense(5)(middle1)
# middle1 = Dense(7)(middle1)

######## output 모델 구성 ########
output1 = Dense(12)(dense1_2)
output1_2 = Dense(24)(output1)
output1_2 = Dense(48)(output1_2)
output1_2 = Dense(24)(output1_2)
output1_2 = Dense(10)(output1_2)
output1_3 = Dense(2, name='output1_3')(output1_2)

output2 = Dense(12)(dense1_2)
output2_2 = Dense(24)(output2)
output2_2 = Dense(48)(output2_2)
output2_2 = Dense(24)(output2_2)
output2_2 = Dense(10)(output2_2)
output2_3 = Dense(2, name='output2_3')(output2_2)

##### 모델 명시 #####
model = Model(inputs=input1,
              outputs=[output1_3, output2_3])

model.summary()
# 자주 실수 : 가끔 서머리는 잘 되는데 피팅돌리면 오류난다. 왜그럴까?
# 제일 많은 경우 : 인풋, 아웃풋 잘못 넣었을 경우가 있다. 체크 잘하자.


# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) # mse 수식은 알고 있자.
model.fit(x1_train,
          [y1_train, y2_train], epochs=100, batch_size=2,
          validation_split=0.25, verbose=2)

# 4. 평가, 예측
loss = model.evaluate(x1_test,
                     [y1_test, y2_test], batch_size=2)

print("model.metrics_names : ", model.metrics_names)
print("loss :", loss)

y1_predict, y2_predict = model.predict(x1_test)
print("============")
print(y1_predict)
print("============")
print(y2_predict)
print("============")


# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
# def로 함수 정의 가로안에 입력되는 것을 넣은 것
# 리턴은 다시 그 값을 되돌려 주는 것. 그래서 RMSE1값은 리턴 값이 들어가는 것이다.

RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE : ", (RMSE1 + RMSE2)/2)
# 캐글이나 해커톤에서 돌려보면서 값이 떨어지고 있다.? 그렇다면 처음 것으로 서브밋을 해야한다.


# R2 구하기
from sklearn.metrics import r2_score
r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)
print("R2_1 : ", r2_1)
print("R2_2 : ", r2_2)
print("R2 : ", (r2_1 + r2_2)/2)