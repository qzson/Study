# 20.05.22_10day_1225~
# 실습
# LSTM 레이어를 5개 엮어서 Dense 결과를 이겨내시오.

'''
튜닝 값
loss: 7.5721e-06 [[79.9553]]
loss: 7.4368e-06 [[79.6716]]
loss: 1.0157e-05 [[79.62378]]

loss: 0.0042 / 84.937546
loss: 5.9646e-04 [[87.89087]]
loss: 1.4579e-04 [[88.59674]]
loss: 7.6402e-05 [[89.78299]]
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
x_predict = array([55,65,75]) # (3,)

print("x.shape:", x.shape) # (13, 3)
print("y.shape:", y.shape) # (13,)

# x = x.reshape(4, 3, 1)                      
x = x.reshape(x.shape[0], x.shape[1], 1) # (13, 3, 1)
print(x.shape)


# 2. 모델구성
input1 = Input(shape=(3,1))
LSTM1 = LSTM(15,return_sequences=True)(input1)
LSTM1 = LSTM(100,return_sequences=True)(LSTM1)    # 명확하진 않지만 input 되는 값이 순차적 data가 아닐 수 있다.
LSTM1 = LSTM(300,return_sequences=True)(LSTM1)    # LSTM을 많이 써도 안좋을 수 있는 이유이다.
LSTM1 = LSTM(100,return_sequences=True)(LSTM1)
LSTM1 = LSTM(15,return_sequences=False)(LSTM1)
output1 = Dense(1, name='output1')(LSTM1)

model = Model(inputs=input1, outputs=output1)

model.summary()


# 3. 훈련
model.compile(optimizer='adam', loss='mse')

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=200, mode='auto')

model.fit(x, y, epochs=1000, batch_size=32, verbose=2,
          callbacks=[early_stopping])

x_predict = x_predict.reshape(1,3,1)
# print(x_predict)

# 4. 예측
y_predict = model.predict(x_predict)
print(y_predict)

# lstm 나가는 값이 순차적인 값이 아니기 때문에
# 그래서 잘나오게 하려면 순차적인 값이 나오도록 해야 결과가 잘 나온다.
# 순차적인 데이터로 나가는 아웃풋이 아니다.