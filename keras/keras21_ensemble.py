# 200518 1030 ~ 앙상블

# 1. 데이터
import numpy as np
x1 = np.array([range(1,101), range(311, 411), range(100)])
y1 = np.array([range(711,811), range(711, 811), range(100)])

x2 = np.array([range(101,201), range(411, 511), range(100,200)])
y2 = np.array([range(501,601), range(711, 811), range(100)]) 

# w와 bias 값이 데이터끼리 같을 필요는 없다..? / 데이터 개수만 같으면 된다.
# 우리가 실질적으로 쓰려는 데이타는 100행 3열 형태다.
# x2,y2 포함 2개 모델 만들고 합칠 수 있다. 1+1 모델

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, shuffle = False,
    train_size = 0.8)

x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, shuffle = False,
    train_size = 0.8)

# x1 트레인 80 바이 3 / x1 테스트는 20 바이 3
# validation_date 쓴다면 train_test_split 함수는 4번 사용해야한다? 맞나

# print(x_train)
# print(x_test)


# 2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

######## 모델 1 ########
input1 = Input(shape=(3,))
dense1_1 = Dense(7, activation='relu', name='bitking1')(input1)
dense1_2 = Dense(5, activation='relu', name='bitking2')(dense1_1)

######## 모델 2 ########
input2 = Input(shape=(3,))
dense2_1 = Dense(8, activation='relu')(input2)
dense2_2 = Dense(4, activation='relu')(dense2_1)

# input1 모델, input2 모델 프레임 구성한 상태 이제 엮어줄 차례
######## 모델 병합 ########
from keras.layers.merge import concatenate
merge1 = concatenate([dense1_2, dense2_2])

# 각 모델의 마지막 layer를 input으로 넣어줌 : list 형태 (2개 이상이면)
# concatenate에서는 param연산이 이루어 지지 않는다.

middle1 = Dense(30)(merge1) # middle 이라는 레이어 구성
middle1 = Dense(5)(middle1) # 딥러닝 구성 이 사이에서 가중치 병합을 유도하는 것?
middle1 = Dense(7)(middle1) # 병합 후 새로운 layer 설정 가능

# 다시 분리해줘야한다? (100,3) 모델 2개 만들고 단순 컨케트네이트 하고 아웃풋으로 나가기 위해 2개의 모델을 구성할 것
# 결과적으로 총 5개의 모델 구성한 것

######## output 모델 구성 ########
output1 = Dense(30)(middle1) # 분리시 상단 레이어의 아웃풋 부분 네임만 써주면 된다. 함수형 모델의 최대 장점
output1_2 = Dense(7)(output1)
output1_3 = Dense(3, name='ss1_3')(output1_2) # 모델 1의 마지막 output

output2 = Dense(25)(middle1)
output2_2 = Dense(5)(output2)
output2_3 = Dense(3, name='ss2_3')(output2_2) # 모델 2의 마지막 output

##### 모델 명시 #####
model = Model(inputs=[input1, input2],
              outputs=[output1_3, output2_3]) # 5개 모델의 시작점 input1, 2

# 서머리 그림 그려서 확인해보고
# 나 보기 편하라고 네이밍을 해줘야한다 ? > 스스로 해볼 문제 / 네임이란 파라미터

model.summary() # 두 모델의 layer 가 번갈아 나온다.

''' summary 구조
인풋 1              인풋 2
dense1_1  ------>  dense2_1 ----> (dense1_2)
dense1_2  ------>  dense2_2
    conc(dense1_2 + dense2_2)
            30
            5
            7
아웃풋1             아웃풋2
output1             output2
output1_2           output2_2
output1_3           output2_3
'''

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train],
          [y1_train, y2_train], epochs=10, batch_size=1,
          validation_split=0.25, verbose=1)
          # 리스트로 묶어 쌍을 만든다.

# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test],
                           [y1_test, y2_test], batch_size=1)
# evaluate 에서 batch_size 를 미기재시, default 값으로 연산.
# 그러므로 fit의 batch_size와 값이 맞지 않을 수 있으므로 통합하는 것이 바람직하다고 판단

print("model.metrics_names : ", model.metrics_names)
# model.metrics_names : evaluate 값으로 무엇이 나오는지 알려준다.
# [(총loss값), (loss 1[아웃풋1]), (loss 2[아웃풋2]), (metrics 1[아웃풋1]), (metrics 2[아웃풋2])]
print("loss :", loss)
# print("mse :", mse) # 필요 없어서 주석처리?

# loss : 모델이 다중 아웃풋을 갖는 경우, 손실의 리스트 혹은 손실의 딕셔너리를 전달하여 각 아웃풋에 각기 다른 손실을 사용할 수 있습니다.
#       따라서 모델에 의해 최소화되는 손실 값은 모든 개별적 손실의 합이 됩니다.

y1_predict, y2_predict = model.predict([x1_test, x2_test])
print("============")
print(y1_predict)
print("============")
print(y2_predict)
print("============")

# x1_test : (20, 3) // x2_test : (20, 3) // y = (20, 3) 2개 나온다.
# y_predict 값은 리스트 형식

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE : ", (RMSE1 + RMSE2)/2)


# R2 구하기
from sklearn.metrics import r2_score
r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)
print("R2_1 : ", r2_1)
print("R2_2 : ", r2_2)
print("R2 : ", (r2_1 + r2_2)/2)

# 병합 과정에서 가중치가 보정 되는데 앙상블 하면서 좋아지는 경우가 있다..?
# 앙상블이 무조건 정답은 아니다.