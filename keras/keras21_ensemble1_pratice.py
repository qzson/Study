# 200518 1030 ~ <결정계수 코드 변경-다른 방식>

# 1. 데이터
import numpy as np
x1 = np.array([range(1,101), range(311, 411), range(100)])
y1 = np.array([range(711,811), range(711, 811), range(100)])

x2 = np.array([range(101,201), range(411, 511), range(100,200)])
y2 = np.array([range(501,601), range(711, 811), range(100)]) # ??

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, shuffle = False,
    train_size = 0.8)

from sklearn.model_selection import train_test_split
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, shuffle = False,
    train_size = 0.8)

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

######## 모델 병합 ########
from keras.layers.merge import concatenate
merge1 = concatenate([dense1_2, dense2_2])

middle1 = Dense(30)(merge1)
middle1 = Dense(5)(middle1)
middle1 = Dense(7)(middle1)

######## output 모델 구성 ########
output1 = Dense(30)(middle1)
output1_2 = Dense(7)(output1)
output1_3 = Dense(3, name='ss1_3')(output1_2)

output2 = Dense(25)(middle1)
output2_2 = Dense(5)(output2)
output2_3 = Dense(3, name='ss2_3')(output2_2)

##### 모델 명시 #####
model = Model(inputs=[input1, input2],
              outputs=[output1_3, output2_3])

model.summary()


# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train],
          [y1_train, y2_train], epochs=10, batch_size=1,
          validation_split=0.25, verbose=1)

# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test],
                           [y1_test, y2_test], batch_size=1)

print("loss :", loss)
# print("mse :", mse)

y1_predict, y2_predict = model.predict([x1_test, x2_test])
print('Predict : ', y1_predict, y2_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
print('RMSE(y1_test) : ', RMSE1)
print('RMSE(y2_test) : ', RMSE2)
print("RMSE : ", (RMSE1+RMSE2)/2)


# R2 구하기
from sklearn.metrics import r2_score
def R2(y_test, y_predict):
    return r2_score(y_test, y_predict)

R2_1 = R2(y1_test, y1_predict)
R2_2 = R2(y2_test, y2_predict)
print('R2(y1_test) : ', R2_1)
print('R2(y2_test) : ', R2_2)
print('R2 : ', (R2_1+R2_2)/2)