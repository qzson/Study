# 20.05.21 1431~
# 성능 비교, 파라미터 토탈 개수파악
# 자원 성능, 무엇이 효율적인지.
'''
keras29_LSTM2 와 SimpleRNN 비교 (같은 튜닝 값, 적은 데이터량)

         정확도                           총 파라미터 수
LSTM2       : 7.9774485     //      Total params: 3,779,581
SimpleRNN   : 7.9450936     //      Total params: 1,344,181
GRU         : 7.983179      //      Total params: 2,967,781

* LSTM vs simpleRNN

  정확도     : 유사  |  자원대비 효율성  : LSTM < SimpleRNN
= 상대적으로 적은 데이터 기준, 효율성은 SimpleRNN 이 좋다.

* simpleRNN vs GRU

  정확도 : < (GRU) |   효율성 : (simpleRNN) >     
= 정확도는 GRU 조금 우세. BUT, 효율성은 simpleRNN 우세. 

'''

from numpy import array
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN


# 1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])    # (4,3)
y = array([4,5,6,7])                            # (4, ) : 스칼라 4, 벡터 1 // (4, ) != (4,1)

print("x.shape:", x.shape)                      # (4,3)
print("y.shape:", y.shape)                      # (4, )

x = x.reshape(x.shape[0], x.shape[1], 1)

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

x_predict = array([5,6,7])            
x_predict = x_predict.reshape(1,3,1)


# 4. 예측
print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)