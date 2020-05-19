# 200519, 1200 ~ / 
# keras14를 Sequential 에서 함수형으로 변경
# earlyStopping 적용
# 튜닝 결과 값 : RMSE = 0.1622, R2 = 0.99997


# 1. 데이터
import numpy as np
x = np.transpose([range(1,101), range(311, 411), range(100)])
y = np.transpose([range(101,201), range(711,811), range(100)])

# print(x.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state = 66, shuffle = True,
    # x, y, shuffle = False,
    train_size = 0.8)

# shuffle
'''
True 가 성능이 더 좋은 이유 ex)
False : 1~100 > 1~80 / 81~100 (훈련범위 안에서만 훈련하게 된다. train 범위 밖은 훈련하지 않으니까)
True : 1~100 데이터 전체에서 섞어서 해주기 때문에
'''

# print(x_train)
# print(x_test)


# 2. 모델구성
from keras.models import Model
from keras.layers import Dense, Input

#### INPUT ####
input1 = Input(shape=(3,))
dense1_1 = Dense(6, activation='relu', name='dense1_1')(input1)
dense1_1 = Dense(9, activation='relu', name='dense1_2')(dense1_1)
dense1_1 = Dense(12, activation='relu', name='dense1_4')(dense1_1)
dense1_2 = Dense(15, activation='relu', name='dense1_5')(dense1_1)

#### OUTPUT ####
output1 = Dense(3)(dense1_2)
output1_1 = Dense(6)(output1)
output1_1 = Dense(9)(output1_1)
output1_1 = Dense(12)(output1_1)
output1_1 = Dense(9)(output1_1)
output1_2 = Dense(3, name='output1_2')(output1_1)

#### MODEL ####
model = Model(inputs = input1,
              outputs = output1_2)

model.summary()


# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 50, mode = 'auto')
model.fit(x_train, y_train, epochs=70, batch_size=1, verbose = 2,
          validation_split=0.25,
          callbacks=[early_stopping])

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test,
                      batch_size = 1)

print("model.metrics_names : ", model.metrics_names) 
print("loss :", loss)

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