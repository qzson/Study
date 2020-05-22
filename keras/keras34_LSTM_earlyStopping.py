# 20.05.22_10day_0900 ~
# 함수형으로 리뉴얼하시오.
# earlyStopping 을 적용하시오.

'''
튜닝 값
loss: 7.5721e-06 [[79.9553]]
loss: 7.4368e-06 [[79.6716]]
loss: 1.0157e-05 [[79.62378]]
'''

from numpy import array
from keras.models import Model
from keras.layers import Dense, LSTM, Input


# 1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = array([50,60,70]) # (3,)

print("x.shape:", x.shape) # (13, 3)
print("y.shape:", y.shape) # (13, ) != (13, 1)
                           #   벡터       행렬

# x = x.reshape(13, 3, 1)                      
x = x.reshape(x.shape[0], x.shape[1], 1) # (13, 3, 1)
print(x.shape)


# 2. 모델구성
input1 = Input(shape=(3,1))
dense1 = LSTM(500, activation='relu', name='dense1')(input1)
dense1 = Dense(250, activation='relu', name='dense2')(dense1)
dense1 = Dense(150, activation='relu', name='dense3')(dense1)
dense1 = Dense(110, activation='relu', name='dense7')(dense1)
dense1 = Dense(60, activation='relu', name='dense8')(dense1)
output1 = Dense(1, name='output1')(dense1)

model = Model(inputs=input1, outputs=output1)

model.summary()


# 3. 훈련
model.compile(optimizer='adam', loss='mse')

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')

model.fit(x, y, epochs=400, batch_size=32, verbose=2,
          callbacks=[early_stopping])

x_predict = x_predict.reshape(1,3,1)     # x값 (13, 3, 1)와 동일한 shape로 만들어 주기 위함
                                         # (1, 3, 1) : 확인 1 * 3 * 1 = 3
# x_predict = x_predict.reshape(1, x_predict.shape[0], 1)

print(x_predict)

# 4. 예측
y_predict = model.predict(x_predict)
print(y_predict)

