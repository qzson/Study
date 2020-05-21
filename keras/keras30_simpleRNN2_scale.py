# 20.05.21 1430~
# 성능 비교, 파라미터 토탈 개수파악
# 자원 성능, 무엇이 효율적인지.
'''
keras29_LSTM3_scale 와 SimpleRNN 비교 (같은 튜닝 값, 많은 데이터량)

         정확도                           총 파라미터 수
LSTM3_scale : 80.0025      //      Total params: 3,779,581
SimpleRNN   : 76.90419     //      Total params: 1,344,181
GRU         : 79.413445    //      Total params: 2,967,781

* LSTM_scale vs simpleRNN

  정확도 : LSTM > simpleRNN  |  자원대비 효율성  : LSTM < SimpleRNN
= scale 기준, 정확도는 LSTM 우세. BUT, 총 파라미터 수는 LSTM 이 더 많다. (약 2~3 배 차이)

* simpleRNN vs GRU

  정확도 : < (GRU) |   효율성 : (simpleRNN) >     
= 정확도는 GRU 조금 우세. BUT, 효율성은 simpleRNN 우세. 

'''

from numpy import array
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

# 1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = array([50,60,70]) # (3,)


print("x.shape:", x.shape) # (13, 3)
print("y.shape:", y.shape) # (13,)

x = x.reshape(x.shape[0], x.shape[1], 1) # (13, 3, 1)

print(x.shape)


# 2. 모델구성
model = Sequential()
# model.add(LSTM(10, activation='relu', input_shape=(3,1)))
model.add(SimpleRNN(900, input_length=3, input_dim=1))
model.add(Dense(450))
model.add(Dense(230))
model.add(Dense(100))
model.add(Dense(1))

model.summary()

# 3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=800, batch_size=32)

x_predict = x_predict.reshape(1,3,1)


# 4. 예측
print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)